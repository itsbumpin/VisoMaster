from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

import torch.nn as nn
import torch.nn.functional as F

from app.processors.models_data import models_dir


@dataclass
class _BackendStatus:
    is_ready: bool = False
    error: Optional[str] = None


class _FiLMBlock(nn.Module):
    """Applies feature-wise affine modulation driven by an age embedding."""

    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(condition_dim, channels)
        self.beta = nn.Linear(condition_dim, channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(condition).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(condition).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class _ResidualConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False)
        self.mod1 = _FiLMBlock(out_channels, condition_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False)
        self.mod2 = _FiLMBlock(out_channels, condition_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.mod1(out, condition)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.mod2(out, condition)
        out = self.act(out + residual)
        return out


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.residual = _ResidualConv(in_channels, out_channels, condition_dim)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.residual(x, condition)
        down = self.down(features)
        return features, down


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.residual = _ResidualConv(out_channels + skip_channels, out_channels, condition_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([up, skip], dim=1)
        return self.residual(merged, condition)


class ConditionalUNet(nn.Module):
    """Conditional UNet mirroring the face_reaging generator."""

    def __init__(
        self,
        in_channels: int = 3,
        condition_channels: int = 2,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_channels

        self.initial = _ResidualConv(in_channels, base_channels, condition_channels)
        self.encoders = nn.ModuleList(
            [
                _EncoderBlock(base_channels, base_channels * 2, condition_channels),
                _EncoderBlock(base_channels * 2, base_channels * 4, condition_channels),
                _EncoderBlock(base_channels * 4, base_channels * 8, condition_channels),
            ]
        )
        self.bottleneck = _ResidualConv(base_channels * 8, base_channels * 8, condition_channels)
        self.decoders = nn.ModuleList(
            [
                _DecoderBlock(base_channels * 8, base_channels * 8, base_channels * 4, condition_channels),
                _DecoderBlock(base_channels * 4, base_channels * 4, base_channels * 2, condition_channels),
                _DecoderBlock(base_channels * 2, base_channels * 2, base_channels, condition_channels),
            ]
        )
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels // 2, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = condition.to(dtype=x.dtype)
        initial = self.initial(x, condition)
        skip_connections = [initial]
        out = initial
        for encoder in self.encoders:
            skip, out = encoder(out, condition)
            skip_connections.append(skip)

        out = self.bottleneck(out, condition)
        for decoder, skip in zip(self.decoders, reversed(skip_connections[:-1])):
            out = decoder(out, skip, condition)

        out = self.final(out)
        return torch.tanh(out)


def _normalise_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    sample_key = next(iter(state_dict))
    if sample_key.startswith("module."):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    if sample_key.startswith("model."):
        return {key[len("model."):]: value for key, value in state_dict.items()}
    return state_dict


class FaceReagingBackend:
    """Wrapper that loads the official face re-aging UNet if available."""

    def __init__(self, device: str | torch.device, model_path: Path | None = None) -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model_path = Path(model_path) if model_path is not None else Path(models_dir) / "face_reaging" / "best_unet_model.pth"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model: Optional[nn.Module] = None
        self.status = _BackendStatus(is_ready=False, error=None)
        self.condition_channels = 2
        self._load()

    @property
    def is_ready(self) -> bool:
        return self.status.is_ready and self.model is not None

    def _load(self) -> None:
        if not self.model_path.exists():
            self.status = _BackendStatus(
                is_ready=False,
                error=f"Missing checkpoint at {self.model_path}",
            )
            return

        try:
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
            self.model.eval()
            self.status = _BackendStatus(is_ready=True, error=None)
            return
        except Exception:  # pragma: no cover - fallback path
            pass

        state: Optional[object] = None
        try:
            state = torch.load(str(self.model_path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(self.model_path), map_location="cpu")
        except Exception as load_error:
            self.status = _BackendStatus(is_ready=False, error=f"Unable to read checkpoint: {load_error}")
            return

        if isinstance(state, nn.Module):
            self.model = state.to(self.device)
            self.model.eval()
            self.status = _BackendStatus(is_ready=True, error=None)
            return

        if isinstance(state, dict):
            candidates = [
                "state_dict",
                "generator",
                "ema",
                "model_state_dict",
                "netG",
                "G",
                "model",
            ]
            state_dict = None
            for candidate in candidates:
                candidate_value = state.get(candidate) if isinstance(state, dict) else None
                if isinstance(candidate_value, dict):
                    state_dict = candidate_value
                    break
            if state_dict is None:
                state_dict = state

            state_dict = _normalise_state_dict(state_dict)
            model = ConditionalUNet(in_channels=3, condition_channels=self.condition_channels)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                self.status = _BackendStatus(
                    is_ready=len(missing) < len(state_dict),
                    error=f"Missing keys: {len(missing)}, unexpected: {len(unexpected)}",
                )
            else:
                self.status = _BackendStatus(is_ready=True, error=None)
            self.model = model.to(self.device)
            self.model.eval()
            return

        self.status = _BackendStatus(
            is_ready=False,
            error=f"Checkpoint format not recognised ({type(state).__name__})",
        )

    def __call__(
        self,
        image: torch.Tensor,
        current_age: float,
        target_age: float,
    ) -> Optional[torch.Tensor]:
        if not self.is_ready or self.model is None:
            return None

        try:
            batched = image.unsqueeze(0) if image.dim() == 3 else image
            if batched.size(0) != 1:
                batched = batched[:1]
            batched = batched.to(device=self.device, dtype=torch.float32)

            condition = torch.tensor(
                [current_age / 100.0, target_age / 100.0],
                dtype=batched.dtype,
                device=batched.device,
            ).view(1, self.condition_channels)

            if isinstance(self.model, ConditionalUNet):
                model_input = batched * 2.0 - 1.0
                output = self.model(model_input, condition)
            else:
                condition_map = condition.view(1, self.condition_channels, 1, 1).expand(
                    batched.size(0), -1, batched.shape[-2], batched.shape[-1]
                )
                model_input = torch.cat([batched * 2.0 - 1.0, condition_map], dim=1)
                output = self.model(model_input)

            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.dim() == 4:
                output = output[0]

            if isinstance(self.model, ConditionalUNet):
                scaled = torch.clamp(output, -1.0, 1.0)
            else:
                if not output.dtype.is_floating_point:
                    output = output.to(torch.float32)
                scaled = torch.tanh(output)

            result = (scaled + 1.0) * 0.5
            return torch.clamp(result, 0.0, 1.0)
        except Exception as runtime_error:  # pragma: no cover - best effort fallback
            self.status = _BackendStatus(is_ready=False, error=str(runtime_error))
            return None
