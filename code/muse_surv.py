"""
MUSE (ICLR'24, "Multimodal Patient Representation Learning with Missing Modalities
and Labels") FAITHFULLY reproduced for histology-genomics SURVIVAL.

EdgeSAGEConv + GNNStack are copied VERBATIM from the official repo (src/core/gnn.py)
so the bipartite-graph + message-passing mechanism is exact. MML is adapted:
  * num_modalities 3 -> 2 (image, gene)
  * classification head -> Cox RISK head (dim 1) + Cox partial-likelihood loss
  * supervised contrastive: "same label" -> same event-time quartile bin (survival
    analog); unsupervised contrastive + edge-drop augmentation kept EXACT.

MUSE's core idea preserved: a MISSING modality = NO edge in the patient-modality
graph (imputation-free). Missing-robustness comes from edge-drop + contrastive.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

HYPAL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
if HYPAL not in sys.path:
    sys.path.insert(0, HYPAL)
from genodistil_cpkf import cox_partial_likelihood_loss  # noqa: E402


# ---------------- VERBATIM from MUSE src/core/gnn.py ----------------
class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, normalize_emb, aggr="mean", **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels, self.out_channels, self.edge_channels = in_channels, out_channels, edge_channels
        self.normalize_emb = normalize_emb
        self.message_lin = nn.Linear(in_channels + edge_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)
        self.message_activation = nn.ReLU()
        self.update_activation = nn.ReLU()

    def forward(self, x, edge_attr, edge_index):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index):
        m_j = torch.cat((x_j, edge_attr), dim=-1)
        return self.message_activation(self.message_lin(m_j))

    def update(self, aggr_out, x):
        aggr_out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x), dim=-1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out


class GNNStack(nn.Module):
    def __init__(self, node_channels, edge_channels, normalize_embs, num_layers, dropout):
        super().__init__()
        self.num_layers, self.dropout = num_layers, dropout
        self.convs = nn.ModuleList(
            [EdgeSAGEConv(node_channels, node_channels, edge_channels, normalize_embs[l]) for l in range(num_layers)])
        self.edge_update_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(node_channels * 2 + edge_channels, edge_channels), nn.ReLU())
            for _ in range(num_layers - 1)])

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        return mlp(torch.cat((x[edge_index[0]], x[edge_index[1]], edge_attr), dim=-1))

    def forward(self, x, edge_attr, edge_index):
        for l, conv in enumerate(self.convs):
            x = conv(x, edge_attr, edge_index)
            if l < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
        return x
# -------------------------------------------------------------------


class FFNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.25):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MUSESurv(nn.Module):
    def __init__(self, img_dim=1536, gene_dim=256, hidden=128, gnn_layers=3,
                 ffn_layers=2, dropout=0.25, normalize_embs=None):
        super().__init__()
        self.M = 2
        self.enc_img = FFNEncoder(img_dim, hidden, hidden, ffn_layers, dropout)
        self.map_img = nn.Linear(hidden, hidden)
        self.enc_gene = FFNEncoder(gene_dim, hidden, hidden, ffn_layers, dropout)
        self.map_gene = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.modality_nodes = nn.Parameter(torch.randn(self.M, hidden))
        ne = [False] * gnn_layers if normalize_embs is None else [bool(int(i)) for i in normalize_embs]
        self.gnn = GNNStack(hidden, hidden, ne, gnn_layers, dropout)
        self.tau = nn.Parameter(torch.tensor(1 / 0.07), requires_grad=False)
        self.cl_projection = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.risk_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, 1))

    # ---- VERBATIM edge-drop augmentation from MUSE ----
    def edgedrop(self, flag):
        n, m = flag.size()
        for i in range(n):
            count_ones = int(flag[i].sum().item())
            if torch.rand(1) < 0.5 or count_ones <= 1:
                continue
            keep_count = torch.randint(1, count_ones, (1,)).item()
            one_indices = (flag[i] == 1).nonzero(as_tuple=True)[0]
            one_indices = one_indices[torch.randperm(one_indices.size(0))]
            flag[i][one_indices[:count_ones - keep_count]] = 0
        return flag

    def unsup_ce_loss(self, zaz_s):
        target = torch.arange(zaz_s.size(0), device=zaz_s.device)
        return (F.cross_entropy(zaz_s, target) + F.cross_entropy(zaz_s.t(), target)) / 2

    def sup_ce_loss(self, z_s, zaz_s, y):
        y = y.view(-1, 1)
        target = (y == y.t()).float()          # same event-time bin (survival analog)
        loss_z = F.binary_cross_entropy_with_logits(z_s, target)
        loss_zaz = F.binary_cross_entropy_with_logits(zaz_s, target)
        loss_zaz_t = F.binary_cross_entropy_with_logits(zaz_s.t(), target)
        return (2 * loss_z + loss_zaz + loss_zaz_t) / 4

    def _encode(self, x_img, x_gene, im, gm):
        a = self.map_img(self.enc_img(x_img)); a = a.clone(); a[im == 0] = 0
        b = self.map_gene(self.enc_gene(x_gene)); b = b.clone(); b[gm == 0] = 0
        return self.drop(a), self.drop(b)

    def _graph(self, x, x_flag):
        """x: (B, M, h), x_flag: (B, M) -> patient reps z (B, h)."""
        B, M, h = x.shape
        patient = torch.ones(B, h, device=x.device)
        nodes = torch.cat([patient, self.modality_nodes], dim=0)
        ei = x_flag.nonzero().t().clone()
        ei[1] += B
        ei = torch.cat([ei, ei.flip([0])], dim=1)
        ea = x[x_flag.bool()].repeat(2, 1)
        z = self.gnn(nodes, ea, ei)
        return z[:B]

    def forward(self, x_img, x_gene, im, gm, t, e, ybin):
        a, b = self._encode(x_img, x_gene, im, gm)
        x = torch.stack([a, b], dim=1)
        x_flag = torch.stack([im, gm], dim=1)
        z = self._graph(x, x_flag)
        # augmented view via edge-drop
        ag_flag = self.edgedrop(x_flag.clone())
        az = self._graph(x, ag_flag)
        u = F.normalize(self.cl_projection(z), dim=-1)
        au = F.normalize(self.cl_projection(az), dim=-1)
        unsup = self.unsup_ce_loss(torch.matmul(u, au.t()) * self.tau)
        sup = self.sup_ce_loss(torch.matmul(u, u.t()) * self.tau,
                               torch.matmul(u, au.t()) * self.tau, ybin)
        risk = self.risk_head(z).squeeze(-1)
        cox = cox_partial_likelihood_loss(risk, t, e)
        return cox + 0.5 * unsup + 0.5 * sup

    @torch.no_grad()
    def inference(self, x_img, x_gene, im, gm):
        a, b = self._encode(x_img, x_gene, im, gm)
        x = torch.stack([a, b], dim=1)
        x_flag = torch.stack([im, gm], dim=1)
        z = self._graph(x, x_flag)
        return self.risk_head(z).squeeze(-1)
