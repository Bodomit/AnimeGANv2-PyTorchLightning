import types
from typing import List

import pytorch_lightning as pl
import torch
import torchvision
from icecream import ic
from torch import nn
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from .losses import DiscriminatorLoss, GeneratorLoss, InitLoss
from .models import Discriminator, Generator


class AnimeGanV2(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.vgg19 = self.get_vgg19()

        self.init_loss = InitLoss(self.vgg19)
        self.generator_loss = GeneratorLoss(self.vgg19)
        self.discriminator_loss = DiscriminatorLoss()

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

    def training_step(self, batch, batch_idx):

        # Get optimizers for each stage.
        optimizers = self.optimizers(use_pl_optimizer=True)
        assert isinstance(optimizers, list)
        init_opt, g_opt, d_opt = optimizers

        real = batch["real"]
        _ = batch["anime"][0]
        anime_gray = batch["anime"][1]
        anime_smooth = batch["anime_smooth"][1]

        generated = self.generator(real)

        # Initilaize the generator on the first epoch.
        if self.current_epoch < 1:
            i_loss = self.init_loss((real, generated))
            init_opt.zero_grad()  # type: ignore
            self.manual_backward(i_loss)
            init_opt.step()
            return

        # Update the generator.
        generated_logit = self.discriminator(generated)
        g_loss = self.generator_loss(real, anime_gray, generated, generated_logit)

        g_opt.zero_grad()  # type: ignore
        self.manual_backward(g_loss)
        g_opt.step()

        # Update the discriminator.
        real_logit = self.discriminator(real)
        anime_gray_logit = self.discriminator(anime_gray)
        smooth_logit = self.discriminator(anime_smooth)

        d_loss = self.discriminator_loss(
            real_logit, anime_gray_logit, generated_logit, smooth_logit
        )

        d_opt.zero_grad()  # type: ignore
        self.manual_backward(d_loss)
        d_opt.step()

    def validation_step(self, batch, batch_idx):
        generated = self.generator(batch)
        return generated

    def validation_epoch_end(self, generated_batches):

        if isinstance(generated_batches, list):
            images = torch.concat(generated_batches, 0)
        elif isinstance(generated_batches, torch.Tensor):
            images = generated_batches
        else:
            raise ValueError

        # Create image grid and log to tensorboard.
        grid_image = to_pil_image(make_grid(images))
        tensorboard = self.logger.experiment  # type: ignore
        tensorboard.add_image("validation", grid_image, self.current_epoch)

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.generator.parameters(), 1e-4),
            torch.optim.Adam(self.generator.parameters(), 8e-5),
            torch.optim.Adam(self.discriminator.parameters(), 16e-5),
        ]

    def configure_callbacks(self):
        return []
