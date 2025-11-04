"""Lightweight StyleGAN2 building blocks used by SAM's pSp model.

The implementation is adapted from the official SAM repository and
rosinality's StyleGAN2 PyTorch port.  It is intentionally kept minimal and
self-contained so that the SAM checkpoint can be materialised without the
original third-party source tree.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _make_kernel(k: Iterable[float]) -> torch.Tensor:
    kernel = torch.tensor(list(k), dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = kernel[:, None] * kernel[None, :]
    kernel /= kernel.sum()
    return kernel


class PixelNorm(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.rsqrt(input.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bias_init: float = 0.0,
        lr_mul: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.full((out_features,), bias_init))
        else:
            self.register_parameter("bias", None)
        self.lr_mul = lr_mul
        self.scale = (1 / math.sqrt(in_features)) * lr_mul

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        return F.linear(input, self.weight * self.scale, bias=bias)


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(input, negative_slope=self.negative_slope) * math.sqrt(2)


class NoiseInjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, image: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channels: int, size: int = 4) -> None:
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channels, size, size))

    def forward(self, batch: int) -> torch.Tensor:
        return self.input.repeat(batch, 1, 1, 1)


class Blur(nn.Module):
    def __init__(self, kernel: Iterable[float], pad: Tuple[int, int]) -> None:
        super().__init__()
        kernel_tensor = _make_kernel(kernel)
        self.register_buffer("kernel", kernel_tensor)
        self.pad = pad

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel[None, None, :, :]
        _, channel, _, _ = input.shape
        kernel = kernel.repeat(channel, 1, 1, 1)
        return F.conv2d(input, kernel, padding=self.pad, groups=channel)


class EqualConv2d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.register_parameter("bias", None)
        self.stride = stride
        self.padding = padding
        fan_in = in_channel * kernel_size * kernel_size
        self.scale = 1 / math.sqrt(fan_in)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
        downsample: bool = False,
        blur_kernel: Iterable[float] = (1, 3, 3, 1),
    ) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample
        self.demodulate = demodulate

        fan_in = in_channel * kernel_size * kernel_size
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        weight = torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)

        self.modulation = EqualLinear(style_dim, in_channel, bias=True)

        if upsample:
            factor = 2
            blur_kernel = [float(k) for k in blur_kernel]
            kernel = _make_kernel(blur_kernel)
            kernel = kernel * (factor ** 2)
            self.register_buffer("blur_kernel", kernel)
        elif downsample:
            factor = 2
            blur_kernel = [float(k) for k in blur_kernel]
            kernel = _make_kernel(blur_kernel)
            kernel = kernel
            self.register_buffer("blur_kernel", kernel)
        else:
            self.register_buffer("blur_kernel", None)

    def forward(
        self,
        input: torch.Tensor,
        style: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.weight * self.scale
        weight = weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4)) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel,
            in_channel,
            self.kernel_size,
            self.kernel_size,
        )

        input = input.view(1, batch * in_channel, height, width)

        if self.upsample:
            input = input.view(batch, in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(2, 3).reshape(
                batch * in_channel,
                self.out_channel,
                self.kernel_size,
                self.kernel_size,
            )
            out = F.conv_transpose2d(
                input,
                weight,
                padding=0,
                stride=2,
            )
            _, _, height, width = out.shape
            if self.blur_kernel is not None:
                kernel = self.blur_kernel[None, None, :, :]
                kernel = kernel.repeat(self.out_channel, 1, 1, 1)
                out = F.conv2d(out, kernel, padding=1, groups=self.out_channel)
            out = out.view(batch, self.out_channel, height, width)
        elif self.downsample:
            input = input.view(batch, in_channel, height, width)
            if self.blur_kernel is not None:
                kernel = self.blur_kernel[None, None, :, :]
                kernel = kernel.repeat(in_channel, 1, 1, 1)
                input = F.conv2d(input, kernel, padding=1, groups=in_channel)
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, stride=2, padding=self.padding, groups=batch)
            out = out.view(batch, self.out_channel, height // 2, width // 2)
        else:
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            out = out.view(batch, self.out_channel, height, width)

        if noise is not None:
            out = out + noise
        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
        )
        self.noise = NoiseInjection(out_channel)
        self.activate = ScaledLeakyReLU()

    def forward(
        self,
        input: torch.Tensor,
        style: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel: int, style_dim: int) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(
        self,
        input: torch.Tensor,
        style: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            out = out + skip

        return out
