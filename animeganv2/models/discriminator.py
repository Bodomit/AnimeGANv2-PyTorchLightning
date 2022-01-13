import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrizations


class ConvSpectralNorm(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 1,
        pad_mode: str = "zero",
        spectral_norm: bool = False,
        bias: bool = False,
        **kwargs,
    ):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        pad = pad_layer[pad_mode](padding)
        conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

        conv_ = parametrizations.spectral_norm(conv) if spectral_norm else conv

        super().__init__(pad, conv_, **kwargs)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        base_n_ch: int = 64,
        n_dis: int = 3,
        spectral_norm: bool = True,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.base_n_ch = base_n_ch
        self.n_dis = n_dis
        self.spectral_norm = spectral_norm

        channels = self.base_n_ch // 2
        prev_channels = channels

        self.conv_0 = ConvSpectralNorm(
            self.in_ch,
            channels,
            spectral_norm=self.spectral_norm,
        )

        self.conv_list = nn.ModuleList()
        for i in range(1, self.n_dis):

            conv_s2 = ConvSpectralNorm(
                prev_channels,
                channels * 2,
                stride=2,
                spectral_norm=self.spectral_norm,
            )

            conv_s1 = ConvSpectralNorm(
                channels * 2,
                channels * 4,
                spectral_norm=self.spectral_norm,
            )

            self.conv_list.add_module(f"conv_s2_{i}", conv_s2)
            self.conv_list.add_module(f"conv_s1_{i}", conv_s1)

            prev_channels = channels * 4
            channels = channels * 2

        self.last_conv = ConvSpectralNorm(
            prev_channels,
            channels * 2,
            spectral_norm=self.spectral_norm,
        )

        self.layer_norm = nn.LayerNorm([channels * 2, 68, 68])

        self.D_logit = ConvSpectralNorm(
            channels * 2, 1, spectral_norm=self.spectral_norm
        )

    def forward(self, input):
        x = self.conv_0(input)
        x = F.leaky_relu(x, 0.2)

        for conv_i in self.conv_list:
            x = conv_i(x)
            x = F.leaky_relu(x, 0.2)

        x = self.last_conv(x)
        x = self.layer_norm(x)
        x = F.leaky_relu(x, 0.2)

        x = self.D_logit(x)
        return x
