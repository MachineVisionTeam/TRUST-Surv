"""
DCMD-Surv — Decomposed Cross-Modal Distillation for imputation-free
missing-modality survival.

KEY IDEA (the ceiling-breaker, from MKD/LCKD MICCAI'23/'25):
  Instead of distilling only the gene HAZARD RANKING into the image head (weak,
  our old Inverse-PAL), we distill at the FEATURE level: each modality is
  projected to a "general" (modality-shared) component, and we pull the image's
  general component toward the gene's general component. The image head then
  reads this gene-aligned general representation -> gene-informed prediction
  from histology alone -> breaks the ~0.74 image-only ceiling.  No reconstruction
  of gene features => imputation-free.

forward(x_img, x_gene, img_present, gene_present)
  -> (h_F, h_I, h_G, G_img, G_gene)
     h_F : fused hazard (both present)          -> uses HGBF backbone
     h_I : image-only hazard (genes missing)    -> reads gene-aligned G_img
     h_G : gene-only hazard (teacher + image missing) -> reads G_gene (real gene)
     G_img, G_gene : general components (for the feature-alignment loss)
"""
import sys
import torch
import torch.nn as nn

HYPAL_PATH = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
if HYPAL_PATH not in sys.path:
    sys.path.insert(0, HYPAL_PATH)
from genodistil_cpkf import GenoDistilCPKF, GenoDistilConfig  # noqa: E402


def _mlp_head(dim, hidden=64, dropout=0.25):
    m = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                      nn.Dropout(dropout), nn.Linear(hidden, 1))
    for lin in (m[0], m[3]):
        nn.init.kaiming_normal_(lin.weight, nonlinearity="linear")
        nn.init.zeros_(lin.bias)
    return m


def _proj(d):
    m = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d))
    for lin in (m[0], m[2]):
        nn.init.kaiming_normal_(lin.weight, nonlinearity="linear")
        nn.init.zeros_(lin.bias)
    return m


class DCMDSurv(nn.Module):
    def __init__(self, img_dim=1536, gene_dim=256, cfg: GenoDistilConfig | None = None):
        super().__init__()
        self.backbone = GenoDistilCPKF(img_dim=img_dim, gene_dim=gene_dim, cfg=cfg)
        d = self.backbone.cfg.fusion_dim

        # general-component projections (modality-shared subspace for distillation)
        self.gen_img = _proj(d)
        self.gen_gene = _proj(d)
        # heads
        self.head_I = _mlp_head(d)   # image, gene-anchored (reads G_img)
        self.head_G = _mlp_head(d)   # gene teacher + image-missing fallback (reads G_gene)
        # absent tokens (imputation-free)
        self.absent_image = nn.Parameter(torch.randn(d) * 0.1)
        self.absent_gene = nn.Parameter(torch.randn(d) * 0.1)

    @property
    def cfg(self):
        return self.backbone.cfg

    def forward(self, x_img, x_gene, img_present, gene_present):
        b = self.backbone
        z_img = b.image_adapter(x_img)
        z_gene = b.gene_adapter(x_gene)
        G_img = self.gen_img(z_img)          # image general (pulled toward gene general)
        G_gene = self.gen_gene(z_gene)       # gene general (teacher; real gene)

        im = img_present.unsqueeze(-1)
        gm = gene_present.unsqueeze(-1)
        z_img_s = im * z_img + (1.0 - im) * self.absent_image
        z_gene_s = gm * z_gene + (1.0 - gm) * self.absent_gene

        fused = b.fusion(z_img_s, z_gene_s)
        h_F = b.cox_head_main(fused).squeeze(-1)
        h_I = self.head_I(G_img).squeeze(-1)     # gene-informed image prediction
        h_G = self.head_G(z_gene).squeeze(-1)    # gene fallback: FULL gene (not just general)
        return h_F, h_I, h_G, G_img, G_gene
