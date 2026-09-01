"""Small synthetic samples for API smoke tests."""

from __future__ import annotations

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
    raise ImportError("Dummy data requires PyTorch") from exc


class DummyOliveFlounderDataset(Dataset):
    """Generate deterministic tensors that follow the dataset contract."""

    def __init__(
        self,
        *,
        size: int = 4,
        image_size: int | tuple[int, int] = (64, 64),
        seed: int = 45,
    ) -> None:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.height, self.width = int(image_size[0]), int(image_size[1])
        if size <= 0 or self.height < 16 or self.width < 16:
            raise ValueError("size must be positive and images must be at least 16x16")
        self.size = int(size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(self.seed + int(index))
        height, width = self.height, self.width
        image = torch.randn((3, height, width), generator=generator)
        fish_region = torch.zeros((height, width), dtype=torch.bool)
        fish_region[height // 4 : 3 * height // 4, width // 8 : 7 * width // 8] = True
        part = torch.zeros((height, width), dtype=torch.long)
        part[fish_region] = 1
        part[height // 3 : 2 * height // 3, width // 8 : width // 4] = 4
        part[height // 3 : 2 * height // 3, 3 * width // 4 : 7 * width // 8] = 3
        part[height // 4 : height // 3, width // 3 : 2 * width // 3] = 2
        semantic = torch.zeros((height, width), dtype=torch.long)
        row = height // 2 + (int(index) % max(1, height // 8))
        semantic[row - 2 : row + 2, width // 2 - 3 : width // 2 + 3] = 1
        symptom_foreground = (semantic > 0).float().unsqueeze(0)
        fish_region_map = fish_region.float().unsqueeze(0)
        unaffected_surface = (fish_region & ~(semantic > 0)).float().unsqueeze(0)
        zero = torch.zeros((1, height, width), dtype=torch.float32)
        direction = torch.full((1, height, width), -1.0)
        x = torch.linspace(0.0, 1.0, width).reshape(1, 1, width).expand(1, height, width)
        direction[fish_region_map > 0] = x[fish_region_map > 0]
        zone = torch.zeros((height, width), dtype=torch.long)
        zone[part == 1] = 1
        zone[part == 4] = 2
        zone[part == 2] = 4
        zone[part == 3] = 7
        return {
            "image": image,
            "semantic_target": semantic,
            "fish_region_target": fish_region_map,
            "part_map_target": part,
            "symptom_foreground_target": symptom_foreground,
            "unaffected_surface_target": unaffected_surface,
            "signed_distance_target": zero.clone(),
            "head_to_tail_direction_target": direction,
            "zone_map_target": zone,
            "semantic_valid": ((semantic > 0) | ~fish_region).float().unsqueeze(0),
            "structure_valid": fish_region_map,
            "positive_evidence_valid": symptom_foreground,
            "unaffected_surface_valid": torch.maximum(
                symptom_foreground,
                unaffected_surface,
            ),
            "boundary_valid": zero.clone(),
            "route_valid": symptom_foreground,
            "file_name": f"synthetic_{index:04d}.png",
            "fish_id": f"synthetic_{index:04d}",
            "image_id": int(index),
        }


def create_dummy_dataloader(
    *,
    size: int = 4,
    image_size: int | tuple[int, int] = (64, 64),
    batch_size: int = 2,
    seed: int = 45,
    shuffle: bool = False,
) -> DataLoader:
    dataset = DummyOliveFlounderDataset(size=size, image_size=image_size, seed=seed)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle))


def build_dummy_dataloaders(
    *,
    num_samples: int = 8,
    image_size: int | tuple[int, int] = (64, 64),
    batch_size: int = 2,
    seed: int = 45,
) -> tuple[DataLoader, DataLoader]:
    train = create_dummy_dataloader(
        size=num_samples,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )
    validation = create_dummy_dataloader(
        size=max(2, num_samples // 2),
        image_size=image_size,
        batch_size=batch_size,
        seed=seed + 10_000,
    )
    return train, validation


def build_dummy_evaluation_dataloaders(
    *,
    num_samples: int = 8,
    image_size: int | tuple[int, int] = (64, 64),
    batch_size: int = 2,
    seed: int = 45,
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic synthetic loaders for both splits."""

    train = create_dummy_dataloader(
        size=num_samples,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )
    validation = create_dummy_dataloader(
        size=max(2, num_samples // 2),
        image_size=image_size,
        batch_size=batch_size,
        seed=seed + 10_000,
        shuffle=False,
    )
    return train, validation
