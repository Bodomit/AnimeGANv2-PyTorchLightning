import types

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torchvision.utils import make_grid

from .losses import DiscriminatorLoss, GeneratorLoss, InitLoss
from .models import Discriminator, Generator


class AnimeGanV2(pl.LightningModule):
    def __init__(self, init_epochs: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.init_epochs = init_epochs

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
            return x

        vgg19.forward = types.MethodType(forward, vgg19)

        # Don't compute grads for feature model.
        vgg19.requires_grad_(False)

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

        # Initilaize the generator on the first n epochs.
        if self.current_epoch < self.init_epochs:
            i_loss = self.init_loss((real, generated))
            init_opt.zero_grad()  # type: ignore
            self.manual_backward(i_loss)
            init_opt.step()
            self.log("i_loss", i_loss)
            return

        # Update the generator.
        generated_logit = self.discriminator(generated)
        g_loss, g_losses = self.generator_loss(
            (real, anime_gray, generated, generated_logit)
        )

        self.log("g_loss", g_loss, prog_bar=True)
        self.log_dict(g_losses)

        g_opt.zero_grad()  # type: ignore
        self.manual_backward(g_loss)
        g_opt.step()

        # Update the discriminator.
        generated = self.generator(real)
        generated_logit = self.discriminator(generated)
        real_logit = self.discriminator(real)
        anime_gray_logit = self.discriminator(anime_gray)
        smooth_logit = self.discriminator(anime_smooth)

        d_loss, d_losses = self.discriminator_loss(
            (real_logit, anime_gray_logit, generated_logit, smooth_logit)
        )

        self.log("d_loss", d_loss, prog_bar=True)
        self.log_dict(d_losses)

        d_opt.zero_grad()  # type: ignore
        self.manual_backward(d_loss)
        d_opt.step()

    def validation_step(self, batch, batch_idx):
        generated = self.generator(batch)
        image_pairs = torch.stack((batch, generated), dim=1)
        image_pairs = torch.flatten(image_pairs, start_dim=0, end_dim=1)

        # Undo linear standardisation
        image_pairs = (image_pairs + 1.0) * 127.5
        image_pairs = torch.clamp(image_pairs, 0, 255)
        image_pairs = image_pairs.to(torch.uint8)

        return image_pairs

    def validation_epoch_end(self, generated_batches):

        if isinstance(generated_batches, list):
            images = torch.concat(generated_batches, 0)
        elif isinstance(generated_batches, torch.Tensor):
            images = generated_batches
        else:
            raise ValueError

        # Create image grid and log to tensorboard.
        grid_image = make_grid(images)
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
