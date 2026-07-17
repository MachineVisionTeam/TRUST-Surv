"""
Flex-MoE (NeurIPS'24) FAITHFUL plain-PyTorch reimplementation (no FastMoE),
adapted to histology-genomics SURVIVAL. Captures Flex-MoE's ACTUAL contribution:

  * Missing-modality bank      : learned embedding per (observed-combo, modality),
                                 replaces a missing modality's token (main.py L82-84).
  * G-Router (generalized)     : NoisyTopKGate + cv_squared load-balancing loss
                                 (Shazeer'17) -> full-modality samples train all
                                 experts with generalized knowledge.
  * S-Router (specialized)     : cross-entropy pushing INCOMPLETE samples (combo!=0)
                                 gate-logits toward the expert == their combo index
                                 (moe_module.py AddtionalNoisyGate.forward).
  * Two-phase schedule         : warm-up = G-router only, then enable S-router.

Combination convention (Flex-MoE uses combo 0 == full-modality):
  both present -> 0 | image-only(gene miss) -> 1 | gene-only(image miss) -> 2
Cox head replaces the classification head. Pooled features -> 1 token/modality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d, dh, dropout=0.1):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, dh), nn.Linear(dh, d)
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class NoisyTopKGate(nn.Module):
    """G-Router (noisy top-k + cv_squared load balance) + S-Router (combo CE)."""
    def __init__(self, d, num_experts, top_k=2):
        super().__init__()
        self.w_gate = nn.Parameter(torch.zeros(d, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(d, num_experts))
        nn.init.normal_(self.w_gate, std=0.01)
        nn.init.normal_(self.w_noise, std=0.01)
        self.E, self.k = num_experts, top_k
        self.loss = torch.zeros(())

    @staticmethod
    def cv_squared(x):
        eps = 1e-10
        if x.numel() <= 1:
            return torch.zeros((), device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, mc=None, srouter=False):
        clean = x @ self.w_gate                                  # (N, E)
        if self.training:
            stddev = F.softplus(x @ self.w_noise) + 1e-2
            logits = clean + torch.randn_like(clean) * stddev
        else:
            logits = clean
        top_logits, top_idx = logits.topk(self.k, dim=1)
        top_gates = F.softmax(top_logits, dim=1)
        gates = torch.zeros_like(logits).scatter(1, top_idx, top_gates)
        # G-Router: load balancing over experts
        loss = self.cv_squared(gates.sum(0))
        # S-Router: incomplete samples (combo != 0) pushed to expert == combo index
        if srouter and mc is not None:
            inc = mc != 0
            if inc.any():
                loss = loss + F.cross_entropy(clean[inc], mc[inc])
        self.loss = loss
        return gates


class MoE(nn.Module):
    def __init__(self, d, dh, num_experts, top_k, dropout=0.1):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d, dh, dropout) for _ in range(num_experts)])
        self.gate = NoisyTopKGate(d, num_experts, top_k)
        self.E = num_experts

    def forward(self, x, mc=None, srouter=False):                # x: (B, T, d)
        B, T, d = x.shape
        xf = x.reshape(-1, d)
        mc_tok = mc.repeat_interleave(T) if mc is not None else None
        gates = self.gate(xf, mc_tok, srouter)                   # (N, E)
        out = torch.zeros_like(xf)
        for e in range(self.E):
            m = gates[:, e] > 0
            if m.any():
                out[m] += gates[m, e:e + 1] * self.experts[e](xf[m])
        return out.reshape(B, T, d)

    def get_loss(self):
        return self.gate.loss


class Attention(nn.Module):
    def __init__(self, d, heads=2, dropout=0.1):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, d = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = self.drop(((q @ k.transpose(-2, -1)) / (self.dh ** 0.5)).softmax(-1))
        return self.proj((a @ v).transpose(1, 2).reshape(B, T, d))


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, num_experts, top_k, sparse, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.sparse = sparse
        if sparse:
            self.mlp = MoE(d, d * 2, num_experts, top_k, dropout)
        else:
            self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d * 2, d))

    def forward(self, x, mc=None, srouter=False):
        x = x + self.drop(self.attn(self.norm1(x)))
        h = self.norm2(x)
        x = x + self.drop(self.mlp(h, mc, srouter) if self.sparse else self.mlp(h))
        return x

    def gate_loss(self):
        return self.mlp.get_loss() if self.sparse else torch.zeros((), device=self.norm1.weight.device)


class FlexMoESurv(nn.Module):
    def __init__(self, img_dim=1536, gene_dim=256, hidden=256, num_layers=2,
                 num_experts=4, top_k=2, heads=2, dropout=0.3):
        super().__init__()
        self.enc_img = nn.Linear(img_dim, hidden)
        self.enc_gene = nn.Linear(gene_dim, hidden)
        self.bank = nn.Parameter(torch.randn(3, 2, hidden) * 0.02)   # [combo, modality, h]
        self.pos = nn.Parameter(torch.zeros(1, 2, hidden))
        layers, sparse = [], True
        for _ in range(num_layers):
            layers.append(EncoderLayer(hidden, heads, num_experts, top_k, sparse, dropout))
            sparse = not sparse
        self.layers = nn.ModuleList(layers)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(True),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_img, x_gene, img_present, gene_present, srouter=False):
        B = x_img.shape[0]
        im, gm = img_present > 0.5, gene_present > 0.5
        # combo 0=both, 1=image-only(gene miss), 2=gene-only(image miss)
        mc = torch.where(im & gm, torch.zeros(B, device=x_img.device, dtype=torch.long),
                         torch.where(im, torch.ones(B, device=x_img.device, dtype=torch.long),
                                     torch.full((B,), 2, device=x_img.device, dtype=torch.long)))
        t_img = torch.where(im.unsqueeze(-1), self.enc_img(x_img), self.bank[mc, 0])
        t_gene = torch.where(gm.unsqueeze(-1), self.enc_gene(x_gene), self.bank[mc, 1])
        x = torch.stack([t_img, t_gene], dim=1) + self.pos
        for layer in self.layers:
            x = layer(x, mc, srouter)
        return self.head(x.reshape(B, -1)).squeeze(-1)

    def gate_loss(self):
        return sum(l.gate_loss() for l in self.layers)
