"""Shared network blocks, tensor helpers, and label constants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: bool = True,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=False) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        dilation: int = 1,
        act: bool = True,
    ):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=k,
            stride=s,
            padding=pad,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.dw_act = nn.SiLU(inplace=False)

        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_act = nn.SiLU(inplace=False) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_act(self.pw_bn(self.pw(x)))
        return x


class ResidualDSConvBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(ch, ch, k=3),
            DepthwiseSeparableConv(ch, ch, k=3, act=False),
        )
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class MultiScaleContextLite(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.b1 = DepthwiseSeparableConv(in_ch, out_ch, k=3, dilation=1)
        self.b2 = DepthwiseSeparableConv(in_ch, out_ch, k=3, dilation=2)
        self.b3 = DepthwiseSeparableConv(in_ch, out_ch, k=3, dilation=4)
        self.fuse = nn.Sequential(
            ConvBNAct(out_ch * 3, out_ch, k=1, s=1, p=0),
            DepthwiseSeparableConv(out_ch, out_ch, k=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class FeatureAdapter(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, k=1, s=1, p=0),
            DepthwiseSeparableConv(out_ch, out_ch, k=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RefineFuseBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, k=1, s=1, p=0),
            DepthwiseSeparableConv(out_ch, out_ch, k=3),
            ResidualDSConvBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LogitHead(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_ch, hidden_ch, k=3),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def safe_log_prob(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(prob.clamp_min(eps))


def sanitize_logits_tensor(x: torch.Tensor, clip: float = 20.0) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=float(clip), neginf=-float(clip))
    return x.clamp(min=-float(clip), max=float(clip))


def safe_binary_prob_from_logits(logits: torch.Tensor, clip: float = 20.0) -> torch.Tensor:
    logits = sanitize_logits_tensor(logits, clip=clip)
    prob = torch.sigmoid(logits)
    prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
    return prob.clamp(0.0, 1.0)


def safe_multiclass_prob_from_logits(
    logits: torch.Tensor,
    dim: int = 1,
    eps: float = 1e-6,
    clip: float = 20.0,
) -> torch.Tensor:
    logits = sanitize_logits_tensor(logits, clip=clip)
    prob = F.softmax(logits, dim=dim)
    prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    den = prob.sum(dim=dim, keepdim=True)
    prob = prob / den.clamp_min(eps)
    if prob.shape[dim] > 0:
        uniform = torch.full_like(prob, 1.0 / float(prob.shape[dim]))
        prob = torch.where((den <= eps).expand_as(prob), uniform, prob)
    return prob


def safe_multiclass_prob_tensor(
    prob: torch.Tensor, dim: int = 1, eps: float = 1e-6
) -> torch.Tensor:
    prob = torch.nan_to_num(prob.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    den = prob.sum(dim=dim, keepdim=True)
    prob = prob / den.clamp_min(eps)
    if prob.shape[dim] > 0:
        uniform = torch.full_like(prob, 1.0 / float(prob.shape[dim]))
        prob = torch.where((den <= eps).expand_as(prob), uniform, prob)
    return prob


def safe_binary_prob_tensor(prob: torch.Tensor) -> torch.Tensor:
    prob = torch.nan_to_num(prob.float(), nan=0.5, posinf=1.0, neginf=0.0)
    return prob.clamp(0.0, 1.0)


def soft_boundary_map(prob: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    boundary = (dx + dy).clamp(0.0, 1.0)
    return boundary


def masked_weighted_mean(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    den = w.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (x * w).sum(dim=(2, 3), keepdim=True) / den


def masked_weighted_std(
    x: torch.Tensor, w: torch.Tensor, mean: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    den = w.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    var = (((x - mean) ** 2) * w).sum(dim=(2, 3), keepdim=True) / den
    return torch.sqrt(var.clamp_min(eps))


def masked_average_pool(
    x: torch.Tensor, mask: torch.Tensor, kernel_size: int = 15, eps: float = 1e-6
) -> torch.Tensor:
    num = F.avg_pool2d(x * mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    den = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2).clamp_min(
        eps
    )
    return num / den


def _weighted_pool(feat: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if weight.shape[-2:] != feat.shape[-2:]:
        weight = F.interpolate(weight, size=feat.shape[-2:], mode="bilinear", align_corners=False)
    den = weight.sum(dim=(2, 3)).clamp_min(eps)
    return (feat * weight).sum(dim=(2, 3)) / den


def _imagenet_denorm(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def _soft_open(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    if k <= 1:
        return x
    erode = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)
    opened = F.max_pool2d(erode, kernel_size=k, stride=1, padding=k // 2)
    return opened.clamp(0.0, 1.0)


def _soft_close(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    if k <= 1:
        return x
    dilate = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    closed = -F.max_pool2d(-dilate, kernel_size=k, stride=1, padding=k // 2)
    return closed.clamp(0.0, 1.0)
