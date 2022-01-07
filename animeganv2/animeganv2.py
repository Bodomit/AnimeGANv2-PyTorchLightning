import types

import pytorch_lightning as pl
import torch
import torchvision
from icecream import ic
from torch import nn

from .models import Discriminator, Generator


class AnimeGanV2(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.vgg19 = self.get_vgg19()

    def get_vgg19(self) -> nn.Module:
        vgg19 = torchvision.models.vgg19(pretrained=True)

        # Monkey-patch to stop forward path through classifier
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        vgg19.forward = types.MethodType(forward, vgg19)

        return vgg19

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch["real"]
        anime = batch["anime"][0]
        anime_gray = batch["anime"][1]
        anime_smooth = batch["anime_smooth"][1]

        x = self.vgg19(real)
        return x

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.generator.parameters(), 1e-4),
            torch.optim.Adam(self.generator.parameters(), 8e-5),
            torch.optim.Adam(self.discriminator.parameters(), 16e-5),
        ]

    def configure_callbacks(self):
        return []
