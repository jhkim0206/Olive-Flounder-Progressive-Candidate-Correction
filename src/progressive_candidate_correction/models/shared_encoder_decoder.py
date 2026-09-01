"""RepViT feature encoder and lightweight shared decoder."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNAct, DepthwiseSeparableConv


class SharedEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "repvit_m1_1.dist_450e_in1k",
        pretrained: bool = True,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        in_chans: int = 3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_chans,
        )
        if not hasattr(self.backbone, "feature_info"):
            raise RuntimeError("Backbone does not expose feature_info.")
        self.out_channels = list(self.backbone.feature_info.channels())
        self.reductions = list(self.backbone.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.backbone(x))


class SharedDecoder(nn.Module):
    def __init__(self, in_channels: list[int], out_ch: int = 128):
        super().__init__()
        assert len(in_channels) == 4, "Expected 4 feature levels from backbone."

        self.lat1 = ConvBNAct(in_channels[0], out_ch, k=1, s=1, p=0)
        self.lat2 = ConvBNAct(in_channels[1], out_ch, k=1, s=1, p=0)
        self.lat3 = ConvBNAct(in_channels[2], out_ch, k=1, s=1, p=0)
        self.lat4 = ConvBNAct(in_channels[3], out_ch, k=1, s=1, p=0)

        self.s1 = DepthwiseSeparableConv(out_ch, out_ch, k=3)
        self.s2 = DepthwiseSeparableConv(out_ch, out_ch, k=3)
        self.s3 = DepthwiseSeparableConv(out_ch, out_ch, k=3)
        self.s4 = DepthwiseSeparableConv(out_ch, out_ch, k=3)

        self.fuse = nn.Sequential(
            ConvBNAct(out_ch * 4, out_ch, k=1, s=1, p=0),
            DepthwiseSeparableConv(out_ch, out_ch, k=3),
        )

    def _up_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        f1, f2, f3, f4 = feats

        p4 = self.s4(self.lat4(f4))
        p3 = self.s3(self.lat3(f3) + self._up_to(p4, f3))
        p2 = self.s2(self.lat2(f2) + self._up_to(p3, f2))
        p1 = self.s1(self.lat1(f1) + self._up_to(p2, f1))

        fused = self.fuse(
            torch.cat(
                [
                    p1,
                    self._up_to(p2, p1),
                    self._up_to(p3, p1),
                    self._up_to(p4, p1),
                ],
                dim=1,
            )
        )

        pyramid = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}
        return fused, pyramid
