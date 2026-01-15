from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoBackbone

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        g = min(groups, out_ch)
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class ColonyNet(nn.Module):
    """SegFormer MiT backbone + FPN neck + 3 heads (sem/center/boundary).

    Backbone is loaded via HuggingFace AutoBackbone (out_indices=(0,1,2,3)).
    """
    def __init__(self, backbone_id: str = "nvidia/mit-b2", fpn_dim: int = 256):
        super().__init__()
        self.backbone = AutoBackbone.from_pretrained(backbone_id, out_indices=(0, 1, 2, 3))
        # Feature maps are a list of tensors [B,C,H,W]
        in_channels = self.backbone.config.hidden_sizes  # list per stage
        if not isinstance(in_channels, (list, tuple)):
            # fallback: some configs may use "hidden_sizes" name
            in_channels = getattr(self.backbone.config, "hidden_sizes", None) or getattr(self.backbone.config, "hidden_size", None)
            raise RuntimeError("Unexpected backbone config; please inspect backbone.config for channel sizes.")

        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_dim, 1) for c in in_channels])
        self.smooth = nn.ModuleList([ConvGNAct(fpn_dim, fpn_dim) for _ in in_channels])

        # Heads
        self.sem_head = nn.Sequential(ConvGNAct(fpn_dim, fpn_dim), nn.Conv2d(fpn_dim, 1, 1))
        self.center_head = nn.Sequential(ConvGNAct(fpn_dim, fpn_dim), nn.Conv2d(fpn_dim, 1, 1))
        self.boundary_head = nn.Sequential(ConvGNAct(fpn_dim, fpn_dim), nn.Conv2d(fpn_dim, 1, 1))

    def forward(self, x):
        # x: B,3,H,W
        out = self.backbone(pixel_values=x, return_dict=True)
        feats = out.feature_maps  # tuple/list of feature maps
        # build FPN (top-down)
        feats = list(feats)  # [C1..C4] low->high stride
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        p = [None] * len(lat)
        p[-1] = lat[-1]
        for i in range(len(lat)-2, -1, -1):
            up = F.interpolate(p[i+1], size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
            p[i] = lat[i] + up
        p = [s(pi) for s, pi in zip(self.smooth, p)]
        # Use highest resolution pyramid level (stride 4)
        f = p[0]
        sem = self.sem_head(f)
        center = self.center_head(f)
        boundary = self.boundary_head(f)
        return {"sem": sem, "center": center, "boundary": boundary}

def set_backbone_trainable(model: ColonyNet, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable
