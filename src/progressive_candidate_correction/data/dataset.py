"""COCO-style adapter for the olive flounder image dataset."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import SEMANTIC_CLASS_NAMES

try:
    import cv2
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
    raise ImportError("OliveFlounderCocoDataset requires the data dependencies") from exc

from .contract import NUM_PART_CLASSES, NUM_SEMANTIC_CLASSES
from .transforms import (
    IMAGE_SIZE,
    build_image_only_transform,
    build_spatial_transform,
    compute_signed_distance_target,
    normalize_image_to_tensor,
)

UNAFFECTED_SURFACE_FISH_REGION_ERODE_RADIUS = 5
UNAFFECTED_SURFACE_SYMPTOM_MARGIN_RADIUS = 7
UNAFFECTED_SURFACE_PART_BOUNDARY_MARGIN_RADIUS = 3
BOUNDARY_BAND_RADIUS = 5
SDF_CLIP_DISTANCE = 16.0


def _read_gray(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return image


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode mask: {path}")
    encoded.tofile(path)


def annotation_mask(annotation: Mapping[str, Any], height: int, width: int) -> np.ndarray:
    """Rasterize one polygon or RLE annotation."""

    segmentation = annotation.get("segmentation", [])
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, list):
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) < 3:
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        return mask

    try:
        from pycocotools import mask as mask_utils
    except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
        raise ImportError("RLE annotations require pycocotools") from exc
    rle = segmentation
    if isinstance(rle.get("counts"), list):
        rle = mask_utils.frPyObjects(rle, height, width)
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.max(axis=2)
    return (decoded > 0).astype(np.uint8)


def load_coco_records(
    root: str | Path,
    split: str,
    cache_dir: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one split and resolve its image and mask paths."""

    root = Path(root).expanduser().resolve()
    split = str(split)
    annotation_path = root / "annotations" / f"{split}_annotations.json"
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = sorted(document["categories"], key=lambda item: int(item["id"]))
    names = tuple(str(item["name"]) for item in categories)
    if names != SEMANTIC_CLASS_NAMES[1:]:
        raise ValueError(f"Unexpected semantic class order: {names}")

    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in document["annotations"]:
        annotations[int(annotation["image_id"])].append(dict(annotation))

    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else root / ".cache" / "semantic_masks"
    )
    records: list[dict[str, Any]] = []
    for image in sorted(document["images"], key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        stem = Path(file_name).stem
        fish_id = str(image.get("fish_id", "")).strip().lower()
        if not fish_id:
            raise ValueError(f"image {image_id} in {annotation_path.name} is missing a fish_id")
        records.append(
            {
                "split": split,
                "image_id": image_id,
                "file_name": file_name,
                "fish_id": fish_id,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "image_path": root / split / "images" / file_name,
                "fish_region_path": root / split / "masks" / f"{stem}.png",
                "part_map_path": root / split / "part_masks" / f"{stem}.png",
                "semantic_cache_path": cache_root / split / f"{stem}.png",
                "annotations": annotations.get(image_id, []),
            }
        )
    return document, records


def build_semantic_mask(record: Mapping[str, Any], *, force: bool = False) -> Path:
    """Build a semantic mask, giving smaller annotations overlap priority."""

    cache_path = Path(record["semantic_cache_path"])
    if cache_path.is_file() and not force:
        return cache_path
    height, width = int(record["height"]), int(record["width"])
    target = np.zeros((height, width), dtype=np.uint8)
    annotations = sorted(
        record["annotations"], key=lambda item: float(item.get("area", 0.0)), reverse=True
    )
    for annotation in annotations:
        class_id = int(annotation["category_id"])
        if not 1 <= class_id < NUM_SEMANTIC_CLASSES:
            raise ValueError(f"Unknown class ID {class_id}")
        target[annotation_mask(annotation, height, width) > 0] = class_id
    _write_png(cache_path, target)
    return cache_path


def _kernel(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _part_boundary(part: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(part, dtype=np.uint8)
    for part_id in range(1, NUM_PART_CLASSES):
        region = (part == part_id).astype(np.uint8)
        boundary |= cv2.morphologyEx(region, cv2.MORPH_GRADIENT, _kernel(1))
    return boundary


def head_to_tail_direction_target(part_map: np.ndarray, fish_region: np.ndarray) -> np.ndarray:
    """Estimate normalized head-to-tail position inside the fish."""

    ys, xs = np.where(fish_region > 0)
    result = np.full(fish_region.shape, -1.0, dtype=np.float32)
    if len(xs) < 2:
        return result

    def centroid(part_id: int) -> np.ndarray | None:
        part_y, part_x = np.where(part_map == part_id)
        if not len(part_x):
            return None
        return np.asarray([part_x.mean(), part_y.mean()], dtype=np.float32)

    head = centroid(4)
    tail = centroid(3)
    points = np.column_stack([xs, ys]).astype(np.float32)
    if head is None or tail is None or float(np.linalg.norm(tail - head)) < 2.0:
        center = points.mean(axis=0)
        centered = points - center[None, :]
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        axis = axes[0].astype(np.float32)
        if tail is not None and float((tail - center) @ axis) < 0:
            axis = -axis
        elif tail is None and head is not None and float((head - center) @ axis) > 0:
            axis = -axis
        elif tail is None and head is None and (axis[0] < 0 or (axis[0] == 0 and axis[1] < 0)):
            axis = -axis
        projection = centered @ axis
        head = center + axis * float(projection.min())
        tail = center + axis * float(projection.max())

    vector = tail - head
    denominator = float(np.dot(vector, vector)) + 1e-6
    xy = np.stack(
        np.meshgrid(np.arange(fish_region.shape[1]), np.arange(fish_region.shape[0])),
        axis=-1,
    ).astype(np.float32)
    projection = ((xy - head) * vector).sum(axis=-1) / denominator
    result[fish_region > 0] = np.clip(projection[fish_region > 0], 0.0, 1.0)
    return result


def zone_map_target(part_map: np.ndarray, fish_region: np.ndarray) -> np.ndarray:
    """Derive body, mouth, tip, middle, and base zones from the Part Map."""

    zone = np.zeros_like(part_map, dtype=np.uint8)
    zone[part_map == 1] = 1
    zone[part_map == 4] = 2
    body_part_dilated = cv2.dilate((part_map == 1).astype(np.uint8), _kernel(3)) > 0
    fish_region_boundary = (
        cv2.morphologyEx(fish_region.astype(np.uint8), cv2.MORPH_GRADIENT, _kernel(2)) > 0
    )
    for part_id, tip_id, middle_id, base_id in ((2, 3, 4, 5), (3, 6, 7, 8)):
        region = part_map == part_id
        base = region & body_part_dilated
        tip = region & fish_region_boundary & ~base
        middle = region & ~base & ~tip
        zone[tip] = tip_id
        zone[middle] = middle_id
        zone[base] = base_id
    return zone


def derived_targets(
    semantic: np.ndarray, fish_region: np.ndarray, part_map: np.ndarray
) -> dict[str, np.ndarray]:
    """Build dense auxiliary targets after spatial augmentation."""

    symptom_foreground = (semantic > 0).astype(np.uint8)
    eroded_fish_region = (
        cv2.erode(
            fish_region.astype(np.uint8),
            _kernel(UNAFFECTED_SURFACE_FISH_REGION_ERODE_RADIUS),
        )
        > 0
    )
    symptom_margin = (
        cv2.dilate(symptom_foreground, _kernel(UNAFFECTED_SURFACE_SYMPTOM_MARGIN_RADIUS)) > 0
    )
    part_boundary_margin = (
        cv2.dilate(
            _part_boundary(part_map),
            _kernel(UNAFFECTED_SURFACE_PART_BOUNDARY_MARGIN_RADIUS),
        )
        > 0
    )
    unaffected_surface = (eroded_fish_region & ~symptom_margin & ~part_boundary_margin).astype(
        np.uint8
    )
    boundary_valid = (
        (cv2.dilate(symptom_foreground, _kernel(BOUNDARY_BAND_RADIUS)) > 0)
        & (cv2.erode(symptom_foreground, _kernel(BOUNDARY_BAND_RADIUS)) == 0)
    ).astype(np.uint8)
    return {
        "symptom_foreground": symptom_foreground,
        "unaffected_surface": unaffected_surface,
        "semantic_valid": ((symptom_foreground > 0) | (fish_region == 0)).astype(np.uint8),
        "structure_valid": fish_region.astype(np.uint8),
        "positive_evidence_valid": symptom_foreground,
        "unaffected_surface_valid": np.maximum(symptom_foreground, unaffected_surface),
        "boundary_valid": boundary_valid,
        "route_valid": symptom_foreground,
        "signed_distance": compute_signed_distance_target(symptom_foreground, SDF_CLIP_DISTANCE),
        "head_to_tail_direction": head_to_tail_direction_target(part_map, fish_region),
        "zone_map": zone_map_target(part_map, fish_region),
    }


class OliveFlounderCocoDataset(Dataset):
    """Load Fish Regions, Part Maps, and COCO symptom annotations."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        training: bool | None = None,
        image_size: tuple[int, int] = IMAGE_SIZE,
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.training = self.split == "train" if training is None else bool(training)
        _, self.records = load_coco_records(self.root, self.split, cache_dir)
        for record in self.records:
            build_semantic_mask(record, force=rebuild_cache)
        self.spatial = build_spatial_transform(self.training, image_size)
        self.color = build_image_only_transform(self.training)

    def __len__(self) -> int:
        return len(self.records)

    def pixel_class_weights(
        self,
        *,
        exponent: float = 0.5,
        background_scale: float = 0.25,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute semantic and part weights from training-mask frequencies."""

        def weights(path_key: str, class_count: int) -> torch.Tensor:
            counts = np.zeros(int(class_count), dtype=np.float64)
            for record in self.records:
                values = _read_gray(Path(record[path_key])).astype(np.int64)
                counts += np.bincount(values.ravel(), minlength=class_count)[:class_count]
            frequency = counts / max(float(counts.sum()), 1.0)
            values = np.power(frequency + 1e-8, -float(exponent))
            values /= max(float(values.mean()), 1e-8)
            values[0] *= float(background_scale)
            return torch.from_numpy(values.astype(np.float32))

        return (
            weights("semantic_cache_path", NUM_SEMANTIC_CLASSES),
            weights("part_map_path", NUM_PART_CLASSES),
        )

    def load_raw(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        return {
            **record,
            "rgb": _read_rgb(Path(record["image_path"])),
            "semantic": _read_gray(Path(record["semantic_cache_path"])).astype(np.uint8),
            "fish_region": (_read_gray(Path(record["fish_region_path"])) > 0).astype(np.uint8),
            "part_map": _read_gray(Path(record["part_map_path"])).astype(np.uint8),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.load_raw(index)
        transformed = self.spatial(
            image=raw["rgb"],
            semantic=raw["semantic"],
            fish_region=raw["fish_region"],
            part_map=raw["part_map"],
        )
        rgb = self.color(image=transformed["image"])["image"]
        semantic = transformed["semantic"].astype(np.uint8)
        fish_region = (transformed["fish_region"] > 0).astype(np.uint8)
        part_map = transformed["part_map"].astype(np.uint8)
        if not np.array_equal(part_map > 0, fish_region > 0):
            raise ValueError(f"Part Map foreground does not match Fish Region: {raw['file_name']}")
        targets = derived_targets(semantic, fish_region, part_map)

        def float_map(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value.astype(np.float32)).unsqueeze(0)

        semantic_tensor = torch.from_numpy(semantic.astype(np.int64))
        part_tensor = torch.from_numpy(part_map.astype(np.int64))
        fish_region_tensor = float_map(fish_region)
        symptom_foreground_tensor = float_map(targets["symptom_foreground"])
        return {
            "image": normalize_image_to_tensor(rgb),
            "semantic_target": semantic_tensor,
            "fish_region_target": fish_region_tensor,
            "part_map_target": part_tensor,
            "symptom_foreground_target": symptom_foreground_tensor,
            "unaffected_surface_target": float_map(targets["unaffected_surface"]),
            "signed_distance_target": float_map(targets["signed_distance"]),
            "head_to_tail_direction_target": float_map(targets["head_to_tail_direction"]),
            "zone_map_target": torch.from_numpy(targets["zone_map"].astype(np.int64)),
            "semantic_valid": float_map(targets["semantic_valid"]),
            "structure_valid": float_map(targets["structure_valid"]),
            "positive_evidence_valid": float_map(targets["positive_evidence_valid"]),
            "unaffected_surface_valid": float_map(targets["unaffected_surface_valid"]),
            "boundary_valid": float_map(targets["boundary_valid"]),
            "route_valid": float_map(targets["route_valid"]),
            "file_name": raw["file_name"],
            "fish_id": raw["fish_id"],
            "image_id": int(raw["image_id"]),
        }


def _worker_seed(_: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_olive_flounder_dataloaders(
    root: str | Path,
    *,
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 45,
    image_size: tuple[int, int] = IMAGE_SIZE,
    training: bool,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = OliveFlounderCocoDataset(
        root,
        "train",
        training=training,
        image_size=image_size,
    )
    val_dataset = OliveFlounderCocoDataset(
        root,
        "val",
        training=False,
        image_size=image_size,
    )
    generator = torch.Generator().manual_seed(int(seed))
    common = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": True,
        "drop_last": False,
        "worker_init_fn": _worker_seed,
        "persistent_workers": bool(num_workers > 0),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=training,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def build_olive_flounder_dataloaders(
    root: str | Path,
    *,
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 45,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Build augmented training and deterministic validation loaders."""

    return _build_olive_flounder_dataloaders(
        root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        image_size=image_size,
        training=True,
    )


def build_olive_flounder_evaluation_dataloaders(
    root: str | Path,
    *,
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 45,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic loaders for both annotation splits."""

    return _build_olive_flounder_dataloaders(
        root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        image_size=image_size,
        training=False,
    )


def validate_dataset(root: str | Path) -> dict[str, int]:
    """Check split counts, fish separation, and mask pairing."""

    counts: dict[str, int] = {}
    fish_ids: dict[str, set[str]] = {}
    for split in ("train", "val"):
        _, records = load_coco_records(root, split)
        fish_ids[split] = {str(record["fish_id"]) for record in records}
        counts[f"{split}_images"] = len(records)
        counts[f"{split}_fish"] = len(fish_ids[split])
        for sample_index, record in enumerate(records):
            for key in ("image_path", "fish_region_path", "part_map_path"):
                if not Path(record[key]).is_file():
                    raise FileNotFoundError(record[key])
            expected_shape = (int(record["height"]), int(record["width"]))
            image = _read_rgb(Path(record["image_path"]))
            fish_region = _read_gray(Path(record["fish_region_path"])) > 0
            part_map = _read_gray(Path(record["part_map_path"]))
            shapes = {image.shape[:2], fish_region.shape, part_map.shape, expected_shape}
            if len(shapes) != 1:
                raise ValueError(f"shape mismatch in {split} sample {sample_index}")
            labels = set(np.unique(part_map).tolist())
            if not labels.issubset(set(range(NUM_PART_CLASSES))):
                raise ValueError(
                    f"invalid part label in {split} sample {sample_index}: {sorted(labels)}"
                )
            if not np.array_equal(part_map > 0, fish_region):
                raise ValueError(
                    f"Part Map foreground does not match Fish Region in {split} "
                    f"sample {sample_index}"
                )
    if fish_ids["train"] & fish_ids["val"]:
        raise ValueError("train and validation fish IDs overlap")
    expected = {
        "train_images": 1169,
        "train_fish": 640,
        "val_images": 292,
        "val_fish": 160,
    }
    if counts != expected:
        raise ValueError(f"dataset counts do not match: {counts}")
    return counts
