"""Image and mask transforms for training and evaluation."""

from __future__ import annotations

import numpy as np

from .contract import BOUNDARY_SDF_CLIP_DISTANCE

IMAGE_SIZE = (384, 384)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
SDF_CLIP_DISTANCE = BOUNDARY_SDF_CLIP_DISTANCE


def _dependencies():
    try:
        import albumentations as A
        import cv2
    except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
        raise ImportError("Transforms require the package's data dependencies") from exc
    return A, cv2


def normalize_image_to_tensor(image_rgb: np.ndarray):
    """Convert an RGB uint8 image to an ImageNet-normalized CHW tensor."""

    try:
        import torch
    except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
        raise ImportError("Image conversion requires PyTorch") from exc
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).float()


def build_spatial_transform(training: bool, image_size: tuple[int, int] = IMAGE_SIZE):
    """Build the joint spatial transform for image, semantic, Fish Region, and Part Map."""

    A, cv2 = _dependencies()
    height, width = (int(image_size[0]), int(image_size[1]))
    operations: list[object] = [A.Resize(height, width)]
    if training:
        operations.append(A.HorizontalFlip(p=0.5))
        options = {
            "shift_limit": 0.03,
            "scale_limit": 0.0,
            "rotate_limit": 15,
            "interpolation": cv2.INTER_LINEAR,
            "border_mode": cv2.BORDER_REFLECT_101,
            "p": 0.75,
        }
        try:
            affine = A.ShiftScaleRotate(**options, fill=0, fill_mask=0)
        except TypeError:  # albumentations 1.x
            affine = A.ShiftScaleRotate(**options, value=0, mask_value=0)
        operations.append(affine)
    return A.Compose(
        operations,
        additional_targets={"semantic": "mask", "fish_region": "mask", "part_map": "mask"},
    )


def build_image_only_transform(training: bool):
    """Build the color-only augmentation applied after spatial transforms."""

    A, _ = _dependencies()
    if not training:
        return A.Compose([])
    return A.Compose(
        [
            A.ColorJitter(
                brightness=0.12,
                contrast=0.10,
                saturation=0.10,
                hue=0.03,
                p=0.50,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.08,
                contrast_limit=0.08,
                p=0.30,
            ),
        ]
    )


def compute_signed_distance_target(
    positive_mask: np.ndarray,
    clip_distance: float = SDF_CLIP_DISTANCE,
) -> np.ndarray:
    """Return a signed distance map, positive inside the target."""

    _, cv2 = _dependencies()
    foreground = (positive_mask > 0).astype(np.uint8)
    if not foreground.any():
        return np.zeros_like(foreground, dtype=np.float32)
    inside = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - foreground, cv2.DIST_L2, 5)
    return np.clip(inside - outside, -float(clip_distance), float(clip_distance)).astype(np.float32)
