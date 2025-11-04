"""Minimal SAM pSp implementation bundled with the project.

This file contains a trimmed-down copy of the architecture definitions used by
Yuval Alaluf's SAM project.  Only the pieces that are required for loading the
aging checkpoint are included here.  The code is adapted from the original
repository to remove external dependencies and to ensure it can be imported
without touching ``sys.path``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import nn

from .stylegan2.generator import Generator
from .stylegan2.layers import EqualLinear


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity()
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv(input)
        skip = self.skip(input)
        return self.activate(out + skip)


class GradualStyleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial: int) -> None:
        super().__init__()
        num_pools = int(math.log2(spatial)) - 2
        channels = in_channels
        modules: List[nn.Module] = []
        for _ in range(max(0, num_pools)):
            modules.append(ConvBlock(channels, channels, downsample=True))
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(channels * 4 * 4, out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.convs(input)
        out = torch.flatten(out, 1)
        return self.linear(out)


class GradualStyleEncoder(nn.Module):
    def __init__(self, n_styles: int = 18, input_nc: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        modules: List[nn.Module] = [
            nn.Conv2d(input_nc, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels = channels[0]
        for out_channels in channels:
            modules.append(ConvBlock(in_channels, out_channels, downsample=True))
            in_channels = out_channels
        self.body = nn.Sequential(*modules)

        style_blocks = []
        spatial = 16
        for _ in range(n_styles):
            style_blocks.append(GradualStyleBlock(in_channels, 512, spatial))
        self.styles = nn.ModuleList(style_blocks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = self.body(input)
        latents = [block(features) for block in self.styles]
        return torch.stack(latents, dim=1)


class Encoder4Editing(GradualStyleEncoder):
    def __init__(self, n_styles: int = 18, input_nc: int = 3) -> None:
        super().__init__(n_styles=n_styles, input_nc=input_nc)

    def forward(
        self,
        input: torch.Tensor,
        latent_mask: Optional[Sequence[int]] = None,
        inject_latent: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        codes = super().forward(input)
        if inject_latent is not None and latent_mask is not None:
            for idx in latent_mask:
                codes[:, idx] = inject_latent[:, idx]
        if alpha is not None and inject_latent is not None and latent_mask is not None:
            for idx in latent_mask:
                codes[:, idx] = alpha * inject_latent[:, idx] + (1 - alpha) * codes[:, idx]
        return codes


# ---------------------------------------------------------------------------
# pSp wrapper
# ---------------------------------------------------------------------------


@dataclass
class PSPOpts:
    input_nc: int = 3
    n_styles: int = 18
    output_size: int = 1024
    start_from_latent_avg: bool = True
    checkpoint_path: Optional[str] = None
    latent_avg: Optional[str] = None
    encoder_type: str = "Encoder4Editing"
    stylegan_weights: Optional[str] = None


class pSp(nn.Module):
    def __init__(self, opts: Mapping[str, Any] | PSPOpts) -> None:
        super().__init__()
        if isinstance(opts, PSPOpts):
            opts_dict = vars(opts)
        else:
            opts_dict = dict(opts)
        self.opts: Dict[str, Any] = opts_dict

        n_styles = int(opts_dict.get("n_styles", 18))
        input_nc = int(opts_dict.get("input_nc", 3))
        output_size = int(opts_dict.get("output_size", 1024))

        encoder_type = opts_dict.get("encoder_type", "Encoder4Editing")
        if encoder_type == "GradualStyleEncoder":
            self.encoder = GradualStyleEncoder(n_styles=n_styles, input_nc=input_nc)
        else:
            self.encoder = Encoder4Editing(n_styles=n_styles, input_nc=input_nc)

        self.decoder = Generator(output_size, 512, 8)
        self.face_pool = nn.AdaptiveAvgPool2d((256, 256))
        self.latent_avg: Optional[torch.Tensor] = None

        latent_avg = opts_dict.get("latent_avg")
        if isinstance(latent_avg, torch.Tensor):
            self.register_buffer("latent_avg_tensor", latent_avg)
            self.latent_avg = latent_avg
        else:
            self.register_buffer("latent_avg_tensor", None)

    def forward(
        self,
        input: torch.Tensor,
        latent_mask: Optional[Sequence[int]] = None,
        inject_latent: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        return_latents: bool = False,
        randomize_noise: bool = True,
        resize: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        latents = self.encoder(input, latent_mask=latent_mask, inject_latent=inject_latent, alpha=alpha)

        if self.latent_avg is not None:
            latents = latents + self.latent_avg[None, :, :]

        if randomize_noise:
            noise = None
        else:
            noise = self.decoder.make_noise()

        result, latent_out = self.decoder(latents, input_is_latent=True, noise=noise, return_latents=True)

        if resize:
            result = self.face_pool(result)

        if return_latents:
            return result, latent_out
        return result, None


class SAMModule(nn.Module):
    """Lightweight wrapper used by :class:`FaceReagingBackend`.

    Parameters
    ----------
    opts:
        The options dictionary bundled with the checkpoint.
    state_dict:
        The state dictionary containing the model weights.
    device:
        Target device for the instantiated module.
    """

    def __init__(
        self,
        opts: Mapping[str, Any],
        state_dict: Mapping[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.model = pSp(opts)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            # Register missing keys to aid debugging without failing initialisation.
            self.register_buffer("_missing_keys", torch.tensor([len(missing)], dtype=torch.int32))
        if unexpected:
            self.register_buffer("_unexpected_keys", torch.tensor([len(unexpected)], dtype=torch.int32))
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        device: Optional[torch.device] = None,
    ) -> "SAMModule":
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
        if state_dict is None:
            raise KeyError("Checkpoint does not contain a state_dict entry")
        opts = checkpoint.get("opts") or checkpoint.get("opt") or {}
        return cls(opts, state_dict, device=device)

    def forward(self, *args: torch.Tensor, **kwargs: Any) -> Any:  # pragma: no cover - delegated
        return self.model(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        if item in {"model", "device", "_modules", "_parameters", "_buffers"}:
            return super().__getattr__(item)
        return getattr(self.model, item)
