from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.processors.models_data import models_dir


@dataclass
class _BackendStatus:
    is_ready: bool = False
    error: Optional[str] = None


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = _ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.conv(x)
        pooled = F.avg_pool2d(residual, kernel_size=2, stride=2, divisor_override=None)
        return residual, pooled


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = _ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.upsample(x)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        concatenated = torch.cat([up, skip], dim=1)
        return self.conv(concatenated)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        condition_channels: int = 2,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        total_in = in_channels + condition_channels
        self.condition_channels = condition_channels

        self.initial = _ConvBlock(total_in, base_channels)
        self.down1 = _DownBlock(base_channels, base_channels * 2)
        self.down2 = _DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = _DownBlock(base_channels * 4, base_channels * 8)
        self.mid = _ConvBlock(base_channels * 8, base_channels * 8)
        self.up3 = _UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = _UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = _UpBlock(base_channels * 2, base_channels, base_channels)
        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.dim() == 2:
            condition = condition.view(condition.size(0), condition.size(1), 1, 1)
        if condition.shape[0] != x.shape[0]:
            condition = condition.expand(x.shape[0], -1, -1, -1)
        condition_map = condition.expand(-1, -1, x.shape[-2], x.shape[-1])
        x = torch.cat([x, condition_map], dim=1)

        enc0 = self.initial(x)
        skip1, pooled1 = self.down1(enc0)
        skip2, pooled2 = self.down2(pooled1)
        skip3, pooled3 = self.down3(pooled2)
        bottleneck = self.mid(pooled3)
        up2 = self.up3(bottleneck, skip3)
        up1 = self.up2(up2, skip2)
        up0 = self.up1(up1, skip1)
        output = self.final(up0)
        return torch.tanh(output)


class FaceReagingBackend:
    """Wrapper that loads the official face re-aging UNet if available.

    When the repository's ``best_unet_model.pth`` checkpoint is present in
    ``model_assets/face_reaging`` the backend will execute that model. If the
    checkpoint is missing or cannot be parsed, the backend simply reports that
    it is unavailable and callers can gracefully fall back to an analytic
    approximation."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model_path = Path(models_dir) / "face_reaging" / "best_unet_model.pth"
        self.model: Optional[nn.Module] = None
        self.status = _BackendStatus(is_ready=False, error=None)
        self.condition_channels = 2
        self._load()

    @property
    def is_ready(self) -> bool:
        return self.status.is_ready and self.model is not None

    def _load(self) -> None:
        if not self.model_path.exists():
            self.status = _BackendStatus(is_ready=False, error="Missing best_unet_model.pth checkpoint")
            return

        try:
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
            self.model.eval()
            self.status = _BackendStatus(is_ready=True, error=None)
            return
        except Exception as jit_error:  # pragma: no cover - fallback path
            last_error = f"torch.jit load failed: {jit_error}"

        state = None
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
            state_dict = state.get("state_dict", state)
            model = ConditionalUNet(in_channels=3, condition_channels=self.condition_channels)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                self.status = _BackendStatus(
                    is_ready=True,
                    error=f"Loaded with missing={len(missing)}, unexpected={len(unexpected)} keys",
                )
            else:
                self.status = _BackendStatus(is_ready=True, error=None)
            self.model = model.to(self.device)
            self.model.eval()
            return

        self.status = _BackendStatus(is_ready=False, error="Checkpoint format not recognised")

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
                model_input = batched * 2 - 1
                output = self.model(model_input, condition)
            else:
                model_input = torch.cat(
                    [batched * 2 - 1, condition.view(1, self.condition_channels, 1, 1).expand(1, -1, batched.shape[-2], batched.shape[-1])],
                    dim=1,
                )
                output = self.model(model_input)

            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.dim() == 4:
                output = output[0]
            result = torch.tanh(output) if output.dtype.is_floating_point else output.to(torch.float32)
            result = (result + 1.0) * 0.5
            return torch.clamp(result, 0.0, 1.0)
        except Exception as runtime_error:  # pragma: no cover - best effort fallback
            self.status = _BackendStatus(is_ready=False, error=str(runtime_error))
            return None
