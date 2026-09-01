"""Baseline models used in the reported comparison."""

from __future__ import annotations

from typing import Any

from ..schema import PART_MAP_ROUTE_INDICES, SEMANTIC_CLASS_NAMES

BASELINE_MODEL_NAMES = ("unet_rgb", "segformer_b0_rgb", "segformer_b0_part")

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # Evaluation and config inspection do not need PyTorch.
    torch = None
    F = None
    nn = None

TORCH_AVAILABLE = torch is not None


def _missing_torch() -> RuntimeError:
    return RuntimeError("baseline models require PyTorch; install the project training extras")


if nn is not None:

    class DoubleConv(nn.Module):
        """Two convolution, batch-normalization, and ReLU blocks."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs: Any) -> Any:
            return self.block(inputs)

    class UNet(nn.Module):
        """Four-level U-Net used for the direct RGB baseline."""

        def __init__(self, in_channels: int = 3, num_classes: int = 9) -> None:
            super().__init__()
            channels = (32, 64, 128, 256)
            self.enc1 = DoubleConv(in_channels, channels[0])
            self.enc2 = DoubleConv(channels[0], channels[1])
            self.enc3 = DoubleConv(channels[1], channels[2])
            self.enc4 = DoubleConv(channels[2], channels[3])
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(channels[3], 512)
            self.up4 = nn.ConvTranspose2d(512, channels[3], 2, stride=2)
            self.dec4 = DoubleConv(channels[3] * 2, channels[3])
            self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
            self.dec3 = DoubleConv(channels[2] * 2, channels[2])
            self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
            self.dec2 = DoubleConv(channels[1] * 2, channels[1])
            self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
            self.dec1 = DoubleConv(channels[0] * 2, channels[0])
            self.head = nn.Conv2d(channels[0], num_classes, 1)

        def forward(self, inputs: Any) -> Any:
            enc1 = self.enc1(inputs)
            enc2 = self.enc2(self.pool(enc1))
            enc3 = self.enc3(self.pool(enc2))
            enc4 = self.enc4(self.pool(enc3))
            bottleneck = self.bottleneck(self.pool(enc4))
            dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
            dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
            dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
            dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
            return self.head(dec1)

else:

    class DoubleConv:  # type: ignore[no-redef]
        """Placeholder shown when PyTorch is unavailable."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise _missing_torch()

    class UNet:  # type: ignore[no-redef]
        """Placeholder shown when PyTorch is unavailable."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise _missing_torch()


def expand_first_rgb_conv(model: Any, in_channels: int) -> str:
    """Expand the first RGB convolution and zero-initialize added channels."""

    if nn is None or torch is None:
        raise _missing_torch()
    if int(in_channels) < 3:
        raise ValueError("in_channels must be at least three")

    first_name = None
    first_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            first_name, first_conv = name, module
            break
    if first_name is None or first_conv is None:
        raise RuntimeError("could not locate the SegFormer RGB patch projection")

    if "." in first_name:
        parent_name, child_name = first_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
    else:
        parent, child_name = model, first_name
    expanded = nn.Conv2d(
        int(in_channels),
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
    )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :3].copy_(first_conv.weight)
        if first_conv.bias is not None:
            expanded.bias.copy_(first_conv.bias)
    setattr(parent, child_name, expanded)
    if hasattr(model, "config"):
        model.config.num_channels = int(in_channels)
    return first_name


def build_segformer_b0(
    *,
    with_part: bool = False,
    pretrained: bool = True,
    model_name: str = "nvidia/mit-b0",
    local_files_only: bool = False,
) -> Any:
    """Build the reported SegFormer-B0 baseline."""

    if torch is None:
        raise _missing_torch()
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as error:
        raise RuntimeError("SegFormer baselines require transformers>=4.40,<5") from error

    id2label = dict(enumerate(SEMANTIC_CLASS_NAMES))
    label2id = {name: class_id for class_id, name in id2label.items()}
    config = SegformerConfig.from_pretrained(
        model_name,
        num_labels=len(SEMANTIC_CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
        local_files_only=local_files_only,
    )
    config.semantic_loss_ignore_index = 255
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        )
    else:
        model = SegformerForSemanticSegmentation(config)
    if with_part:
        expand_first_rgb_conv(model, 3 + len(PART_MAP_ROUTE_INDICES))
    return model


def build_baseline(name: str, **kwargs: Any) -> Any:
    """Build one of the three reported baselines."""

    if name == "unet_rgb":
        return UNet(**kwargs)
    if name == "segformer_b0_rgb":
        return build_segformer_b0(with_part=False, **kwargs)
    if name == "segformer_b0_part":
        return build_segformer_b0(with_part=True, **kwargs)
    raise KeyError(f"unknown baseline: {name}")


def baseline_input(name: str, image: Any, part_map: Any | None = None) -> Any:
    """Prepare RGB or RGB-plus-part input for a baseline."""

    if torch is None:
        raise _missing_torch()
    if name != "segformer_b0_part":
        return image
    if part_map is None:
        raise ValueError("segformer_b0_part requires a Part Map")
    if image.ndim != 4 or part_map.ndim != 3:
        raise ValueError("expected image [B,C,H,W] and Part Map [B,H,W]")
    one_hot = torch.stack(
        [(part_map == part_id).to(dtype=image.dtype) for part_id in PART_MAP_ROUTE_INDICES],
        dim=1,
    )
    return torch.cat([image, one_hot], dim=1)


def forward_baseline(name: str, model: Any, image: Any, part_map: Any | None = None) -> Any:
    """Return logits at the input resolution."""

    if F is None:
        raise _missing_torch()
    inputs = baseline_input(name, image, part_map)
    logits = model(inputs) if name == "unet_rgb" else model(pixel_values=inputs).logits
    if logits.shape[-2:] != image.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return logits


__all__ = [
    "BASELINE_MODEL_NAMES",
    "TORCH_AVAILABLE",
    "DoubleConv",
    "UNet",
    "baseline_input",
    "build_baseline",
    "build_segformer_b0",
    "expand_first_rgb_conv",
    "forward_baseline",
]
