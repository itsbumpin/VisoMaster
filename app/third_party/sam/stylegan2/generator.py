"""StyleGAN2 generator used by the SAM pSp architecture."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from .layers import ConstantInput, PixelNorm, ScaledLeakyReLU, StyledConv, ToRGB, EqualLinear


class Generator(nn.Module):
    def __init__(
        self,
        size: int,
        style_dim: int,
        n_mlp: int,
        channel_multiplier: int = 2,
    ) -> None:
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        self.num_layers = int(math.log2(size) * 2 - 2)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        layers = [PixelNorm()]
        for _ in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=0.01))
            layers.append(ScaledLeakyReLU())
        self.style = nn.Sequential(*layers)

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim)

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        log_size = int(math.log2(size))
        for layer_idx in range(self.num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        in_channel = self.channels[4]
        for i in range(3, log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

    def make_noise(self) -> List[torch.Tensor]:
        noises = []
        for layer_idx in range(self.num_layers):
            res = layer_idx // 2 + 2
            shape = (1, 1, 2 ** res, 2 ** res)
            noises.append(torch.randn(*shape, device=self.input.input.device))
        return noises

    def mean_latent(self, n_latent: int) -> torch.Tensor:
        latent = torch.randn(n_latent, self.style_dim, device=self.input.input.device)
        latent = self.style(latent)
        return latent.mean(0, keepdim=True)

    def forward(
        self,
        styles: List[torch.Tensor] | torch.Tensor,
        return_latents: bool = False,
        inject_index: Optional[int] = None,
        truncation: float = 1.0,
        truncation_latent: Optional[torch.Tensor] = None,
        input_is_latent: bool = False,
        noise: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(styles, list):
            styles = [styles]

        if noise is None:
            noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if truncation < 1.0 and truncation_latent is not None:
            styles = [
                truncation_latent + truncation * (style - truncation_latent)
                for style in styles
            ]

        if len(styles) == 1:
            styles = styles * 2

        latent = torch.stack(styles, 1)

        batch = styles[0].shape[0]
        if inject_index is None:
            inject_index = self.num_layers

        latent = latent[:, 0, :].unsqueeze(1).repeat(1, self.num_layers, 1)

        out = self.input(batch)
        noise_iter = iter(noise)

        out = self.conv1(out, latent[:, 0, :], next(noise_iter, None))
        skip = self.to_rgb1(out, latent[:, 1, :])

        layer = 1
        for conv1, conv2, to_rgb in zip(
            self.convs[::2],
            self.convs[1::2],
            self.to_rgbs,
        ):
            out = conv1(out, latent[:, layer, :], next(noise_iter, None))
            layer += 1
            out = conv2(out, latent[:, layer, :], next(noise_iter, None))
            layer += 1
            skip = to_rgb(out, latent[:, layer - 1, :], skip)

        image = skip

        if return_latents:
            return image, latent
        return image, None
