import os
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset

MAX_WORKERS = 16


def get_n_workers():
    n_cpu = os.cpu_count()
    assert n_cpu
    return min(n_cpu, MAX_WORKERS)


class BasicImageDataset(Dataset):
    def __init__(self, paths: Set[str], transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.paths = list(sorted(paths))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        image = torchvision.io.read_image(path)
        image = image / 127.5 - 1.0

        if self.transform is not None:
            image = self.transform(image)

        return image


class UnlabledImageDataset(BasicImageDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        valid_extentions: Set[str] = set([".png", ".jpg"]),
    ) -> None:

        self.valid_extentions = valid_extentions
        self.image_paths = self.read_image_paths(root)

        super().__init__(
            self.image_paths, transform=transform,
        )

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
    def __init__(
        self,
        real_root: str,
        anime_root: str,
        batch_size: int,
        val_batch_size: int,
        val_set_ratio: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.val_set_ratio = val_set_ratio

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

        real_train_paths, real_val_paths = self.train_test_split(self.real_dataset)
        self.real_train_dataset = BasicImageDataset(real_train_paths, base_transforms)
        self.real_val_dataset = BasicImageDataset(real_val_paths, base_transforms)

        assert 0 not in [
            len(self.real_dataset),
            len(self.anime_dataset),
            len(self.anime_smooth_dataset),
        ]

    def train_test_split(
        self, dataset: UnlabledImageDataset
    ) -> Tuple[Set[str], Set[str]]:
        all_paths = set(dataset.paths)
        val_set_count = int(len(all_paths) * self.val_set_ratio)
        val_set_paths = set(random.sample(all_paths, val_set_count))
        train_set_paths = all_paths - val_set_paths

        return train_set_paths, val_set_paths

    def train_dataloader(self):
        return {
            "real": DataLoader(
                self.real_train_dataset,
                self.batch_size,
                shuffle=True,
                num_workers=get_n_workers(),
                drop_last=True,
            ),
            "anime": DataLoader(
                self.anime_dataset,
                self.batch_size,
                shuffle=True,
                num_workers=get_n_workers(),
                drop_last=True,
            ),
            "anime_smooth": DataLoader(
                self.anime_smooth_dataset,
                self.batch_size,
                shuffle=True,
                num_workers=get_n_workers(),
                drop_last=True,
            ),
        }

    def val_dataloader(self):
        n_cpus = os.cpu_count()
        assert n_cpus
        return DataLoader(
            self.real_val_dataset,
            self.val_batch_size,
            shuffle=False,
            num_workers=get_n_workers(),
        )
