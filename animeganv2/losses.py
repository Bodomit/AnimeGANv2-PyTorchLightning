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


def style_loss(style, generated):
    return F.l1_loss(gram(style), gram(generated))


def colour_loss(real, generated):
    real = rgb_to_ycbcr(real)
    generated = rgb_to_ycbcr(generated)

    return (
        F.l1_loss(real[:, 0, :, :], generated[:, 0, :, :])
        + F.huber_loss(real[:, 0, :, :], generated[:, 1, :, :])
        + F.huber_loss(real[:, 0, :, :], generated[:, 2, :, :])
    )


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        feature_model: nn.Module,
        adversarial_weight: float = 300,
        content_weight: float = 1.5,
        style_weight: float = 3.0,
        colour_weight: float = 10,
    ) -> None:
        super().__init__()
        self.feature_model = feature_model
        self.adversarial_weight = adversarial_weight
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.colour_weight = colour_weight

    def forward(self, input):
        real, gray, generated, generated_logit = input
        real_features = self.feature_model(real)
        gray_features = self.feature_model(gray)
        generated_features = self.feature_model(generated)

        adv_loss = torch.mean(torch.square(generated_logit - 1.0))
        con_loss = content_loss(real_features, generated_features)
        sty_loss = style_loss(real_features, gray_features)
        col_loss = colour_loss(real, generated)

        total_loss = (
            adv_loss * self.adversarial_weight
            + con_loss * self.content_weight
            + sty_loss * self.style_weight
            + col_loss * self.colour_weight
        )

        return total_loss


class InitLoss(nn.Module):
    def __init__(self, feature_model: nn.Module, content_weight: float = 1.5) -> None:
        super().__init__()
        self.feature_model = feature_model
        self.content_weight = content_weight

    def forward(self, input):
        real, generated = input
        real_features = self.feature_model(real)
        generated_features = self.feature_model(generated)
        con_loss = content_loss(real_features, generated_features)
        return con_loss * self.content_weight


class DiscriminatorLoss(nn.Module):
    def forward(self, input):
        real_logit, gray_logit, generated_logit, smooth_logit = input
        real_loss = torch.mean(torch.square(real_logit - 1.0))
        gray_loss = torch.mean(torch.square(gray_logit))
        fake_loss = torch.mean(torch.square(generated_logit))
        smooth_loss = torch.mean(torch.square(smooth_logit))

        loss = real_loss + gray_loss + fake_loss + 0.1 * smooth_loss
        return loss
