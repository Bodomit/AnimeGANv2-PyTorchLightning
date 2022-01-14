from typing import Tuple

import torch
import torch.nn.functional as F
from kornia.color.ycbcr import rgb_to_ycbcr
from torch import nn


def gram(x: torch.Tensor):
    # TODO Check if the CHW or HWC order matters here.
    shape_x = x.shape
    b = shape_x[0]
    c = shape_x[3]
    x = torch.reshape(x, [b, -1, c])
    return (torch.transpose(x, 1, 2) @ x) / (x.numel() // b)


def content_loss(x1_features: torch.Tensor, x2_features: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x1_features, x2_features)


def style_loss(style: torch.Tensor, generated: torch.Tensor):
    return F.l1_loss(gram(style), gram(generated))


def colour_loss(real: torch.Tensor, generated: torch.Tensor):
    real = rgb_to_ycbcr(real)
    generated = rgb_to_ycbcr(generated)

    return (
        F.l1_loss(real[:, 0, :, :], generated[:, 0, :, :])
        + F.huber_loss(real[:, 0, :, :], generated[:, 1, :, :])
        + F.huber_loss(real[:, 0, :, :], generated[:, 2, :, :])
    )


def l2_loss(x: torch.Tensor):
    """
    Matches the equivilent Tensorflow function used in the original AnimeGANv2.
    https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    """
    return (x ** 2).sum() / 2


def total_variation_loss(generated: torch.Tensor):
    # https://en.wikipedia.org/wiki/Total_variation_denoising
    dh = generated[:, :, :-1, :] - generated[:, :, 1:, :]
    dw = generated[:, :, :, :-1] - generated[:, :, :, 1:]
    size_dh = float(dh.numel())
    size_dw = float(dw.numel())

    return l2_loss(dh) / size_dh + l2_loss(dw) / size_dw


def get_features(feature_model: nn.Module, *tensors: torch.Tensor):
    assert all([t.shape == tensors[0].shape for t in tensors])
    stacked_tensors = torch.vstack(tensors)
    features = feature_model(stacked_tensors)
    return torch.tensor_split(features, len(tensors), dim=0)


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        feature_model: nn.Module,
        adversarial_weight: float = 300,
        content_weight: float = 1.5,
        style_weight: float = 3.0,
        colour_weight: float = 10,
        tv_loss_weight: float = 1,
    ) -> None:
        super().__init__()
        self.feature_model = feature_model
        self.adversarial_weight = adversarial_weight
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.colour_weight = colour_weight
        self.tv_loss_weight = tv_loss_weight

    def forward(self, input):
        real, gray, generated, generated_logit = input
        real_features, gray_features, generated_features = get_features(
            self.feature_model, real, gray, generated
        )

        adv_loss = torch.mean(torch.square(generated_logit - 1.0))
        con_loss = content_loss(real_features, generated_features)
        sty_loss = style_loss(real_features, gray_features)
        col_loss = colour_loss(real, generated)
        tv_loss = total_variation_loss(generated)

        total_loss = (
            adv_loss * self.adversarial_weight
            + con_loss * self.content_weight
            + sty_loss * self.style_weight
            + col_loss * self.colour_weight
            + tv_loss * self.tv_loss_weight
        )

        losses = {
            "g_losses/adv": adv_loss,
            "g_losses/con": con_loss,
            "g_losses/sty": sty_loss,
            "g_losses/col": col_loss,
            "g_losses/tv": tv_loss,
        }

        return total_loss, losses


class InitLoss(nn.Module):
    def __init__(self, feature_model: nn.Module, content_weight: float = 1.5) -> None:
        super().__init__()
        self.feature_model = feature_model
        self.content_weight = content_weight

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]):
        real, generated = input
        real_features, generated_features = get_features(
            self.feature_model, real, generated
        )
        con_loss = content_loss(real_features, generated_features)
        return con_loss * self.content_weight


class DiscriminatorLoss(nn.Module):
    def __init__(
        self,
        real_weight: float = 1.7,
        fake_weight: float = 1.7,
        gray_weight: float = 1.7,
        smooth_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.real_weight = real_weight
        self.fake_weight = fake_weight
        self.gray_weight = gray_weight
        self.smooth_weight = smooth_weight

    def forward(self, input):
        real_logit, gray_logit, generated_logit, smooth_logit = input
        real_loss = torch.mean(torch.square(real_logit - 1.0))
        gray_loss = torch.mean(torch.square(gray_logit))
        fake_loss = torch.mean(torch.square(generated_logit))
        smooth_loss = torch.mean(torch.square(smooth_logit))

        loss = (
            real_loss * self.real_weight
            + gray_loss * self.gray_weight
            + fake_loss * self.fake_weight
            + smooth_loss * self.smooth_weight
        )

        losses = {
            "d_losses/real": real_loss,
            "d_losses/gray": gray_loss,
            "d_losses/fake": fake_loss,
            "d_losses/smooth": smooth_loss,
        }

        return loss, losses
