"""
ShaSpec (CVPR'23, Shared-Specific feature modelling for missing modality)
faithfully reimplemented for histology-genomics SURVIVAL on feature vectors.

Core mechanism preserved (from DualNet_SS.py + the paper):
  * SHARED encoder (weights shared across modalities) -> modality-agnostic shared feat
  * per-modality SPECIFIC encoders -> modality-specific feat
  * CompositionalLayer: fuse via RESIDUAL learning  fused = shared + MLP([shared,specific])
  * DOMAIN classification on specific feats (makes them modality-discriminative)
  * distribution ALIGNMENT of shared feats across modalities
  * MISSING modality: reuse the shared feat from present modality + a learned
    "missing specific" token (imputation-free — no raw reconstruction)
Head/loss: Cox (matches DCMD/Flex-MoE/MUSE), so IBS via the same Breslow pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(i, o, h=None, dropout=0.25):
    h = h or o
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(h, o))


class CompositionalLayer(nn.Module):
    """fused = shared + residual, residual = conv([shared, specific]) (ShaSpec L312-341)."""
    def __init__(self, d, dropout=0.25):
        super().__init__()
        self.conv = _mlp(2 * d, d, dropout=dropout)

    def forward(self, shared, specific):
        residual = self.conv(torch.cat([shared, specific], dim=-1))
        return shared + residual


class ShaSpecSurv(nn.Module):
    def __init__(self, img_dim=1536, gene_dim=256, hidden=256, dropout=0.25):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden)
        self.gene_proj = nn.Linear(gene_dim, hidden)
        self.shared_enc = _mlp(hidden, hidden, dropout=dropout)          # SHARED across modalities
        self.spec_img = _mlp(hidden, hidden, dropout=dropout)
        self.spec_gene = _mlp(hidden, hidden, dropout=dropout)
        self.compos = CompositionalLayer(hidden, dropout)               # shared across modalities
        self.missing_spec_img = nn.Parameter(torch.randn(hidden) * 0.02)
        self.missing_spec_gene = nn.Parameter(torch.randn(hidden) * 0.02)
        self.domain_clf = nn.Linear(hidden, 2)                          # specific -> modality id
        self.cox_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(inplace=True),
                                      nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_img, x_gene, img_present, gene_present):
        im = (img_present > 0.5).unsqueeze(-1)
        gm = (gene_present > 0.5).unsqueeze(-1)
        pi = self.img_proj(x_img)
        pg = self.gene_proj(x_gene)
        shared_i = self.shared_enc(pi)      # shared from image
        shared_g = self.shared_enc(pg)      # shared from gene (same encoder)
        spec_i = self.spec_img(pi)
        spec_g = self.spec_gene(pg)
        # MISSING: reuse the other modality's shared + learned missing-specific token
        shared_i_u = torch.where(im, shared_i, shared_g)
        shared_g_u = torch.where(gm, shared_g, shared_i)
        spec_i_u = torch.where(im, spec_i, self.missing_spec_img.expand_as(spec_i))
        spec_g_u = torch.where(gm, spec_g, self.missing_spec_gene.expand_as(spec_g))
        fused_i = self.compos(shared_i_u, spec_i_u)
        fused_g = self.compos(shared_g_u, spec_g_u)
        risk = self.cox_head(torch.cat([fused_i, fused_g], dim=-1)).squeeze(-1)
        return risk, shared_i, shared_g, spec_i, spec_g

    def aux_losses(self, shared_i, shared_g, spec_i, spec_g, both_mask):
        """alignment(shared) + domain-classification(specific)."""
        dev = shared_i.device
        if both_mask.sum() >= 2:
            align = F.mse_loss(F.normalize(shared_i[both_mask], dim=1),
                               F.normalize(shared_g[both_mask], dim=1))
        else:
            align = torch.zeros((), device=dev)
        # domain: image-specific -> class 0, gene-specific -> class 1
        li = self.domain_clf(spec_i); lg = self.domain_clf(spec_g)
        y0 = torch.zeros(spec_i.size(0), dtype=torch.long, device=dev)
        y1 = torch.ones(spec_g.size(0), dtype=torch.long, device=dev)
        domain = 0.5 * (F.cross_entropy(li, y0) + F.cross_entropy(lg, y1))
        return align, domain
