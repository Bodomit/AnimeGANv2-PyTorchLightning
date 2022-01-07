import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset


class UnlabledImageDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        valid_extentions: Set[str] = set([".png", ".jpg"]),
    ) -> None:
        super().__init__(
            root,
            transform=transform,
        )

        self.valid_extentions = valid_extentions
        self.image_paths = list(sorted(self.read_image_paths(root)))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]
        image = torchvision.io.read_image(path)
        image = image / 127.5 - 1.0

        if self.transform is not None:
            image = self.transform(image)

        return image

    def read_image_paths(self, root: str) -> Set[str]:
        image_paths = sorted(
            os.path.join(root, entry.name) for entry in os.scandir(root)
        )
        valid_image_paths = [
            path for path in image_paths if self.is_valid_image_path(path)
        ]

        return set(valid_image_paths)

    def is_valid_image_path(self, path):
        try:
            return os.path.splitext(path)[1] in self.valid_extentions
        except IndexError:
            return False


class WithGrayscaleTransform(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grayscale_transform = transforms.Grayscale(3)

    def __call__(self, x):
        x_grayscale = self.grayscale_transform(x)
        return (x, x_grayscale)


class AnimeGanDataModule(pl.LightningDataModule):
    def __init__(self, real_root: str, anime_root: str, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

        base_transforms = transforms.Compose([transforms.Resize((256, 256))])
        anime_transforms = transforms.Compose(
            [base_transforms, WithGrayscaleTransform()]
        )

        self.real_dataset = UnlabledImageDataset(real_root, base_transforms)
        self.anime_dataset = UnlabledImageDataset(
            os.path.join(anime_root, "style"), anime_transforms
        )
        self.anime_smooth_dataset = UnlabledImageDataset(
            os.path.join(anime_root, "smooth"), anime_transforms
        )

        assert 0 not in [
            len(self.real_dataset),
            len(self.anime_dataset),
            len(self.anime_smooth_dataset),
        ]

    def train_dataloader(self):
        return {
            "real": DataLoader(self.real_dataset, self.batch_size, shuffle=True),
            "anime": DataLoader(self.anime_dataset, self.batch_size, shuffle=True),
            "anime_smooth": DataLoader(
                self.anime_smooth_dataset, self.batch_size, shuffle=True
            ),
        }
