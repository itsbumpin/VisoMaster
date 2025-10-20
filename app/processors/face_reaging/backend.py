from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from app.processors.models_data import models_dir


@dataclass
class _BackendStatus:
    is_ready: bool = False
    error: Optional[str] = None


class _ResidualBlock(nn.Module):
    """Standard ResNet block used by the Face Aging and De-aging GAN."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FaceAgingGenerator(nn.Module):
    """ResNet-based generator mirroring the Gayathry-CB implementation."""

    def __init__(self, input_channels: int, output_channels: int = 3, residual_blocks: int = 9) -> None:
        super().__init__()
        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, kernel_size=7, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features, affine=True, track_running_stats=False),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(residual_blocks):
            model.append(_ResidualBlock(in_features))

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features, affine=True, track_running_stats=False),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, image: torch.Tensor, condition: Optional[torch.Tensor]) -> torch.Tensor:
        if condition is not None:
            if condition.dim() == 2:
                condition = condition[:, :, None, None]
            condition = condition.expand(-1, -1, image.shape[-2], image.shape[-1])
            image = torch.cat([image, condition], dim=1)
        return self.model(image)


def _normalise_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    sample_key = next(iter(state_dict))
    if sample_key.startswith("module."):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    if sample_key.startswith("model."):
        return {key[len("model."):]: value for key, value in state_dict.items()}
    return state_dict


def _find_first_weight_key(state_dict: Dict[str, torch.Tensor], suffixes: Iterable[str]) -> Optional[str]:
    for suffix in suffixes:
        for key in state_dict:
            if key.endswith(suffix):
                return key
    return None


class FaceReagingBackend:
    """Wrapper that loads the Face Aging and De-aging GAN generator."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model_dir = Path(models_dir) / "face_reaging"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path: Optional[Path] = None
        self.model: Optional[FaceAgingGenerator] = None
        self.status = _BackendStatus(is_ready=False, error=None)
        self.label_channels = 0
        self._resolve_checkpoint()
        self._load()

    @property
    def is_ready(self) -> bool:
        return self.status.is_ready and self.model is not None

    def _resolve_checkpoint(self) -> None:
        candidates = [
            "fad_gan_generator.pth",
            "fad_gan.pt",
            "face_aging_gan.pth",
            "face_aging_generator.pth",
            "G.pth",
        ]
        for candidate in candidates:
            path = self.model_dir / candidate
            if path.exists():
                self.model_path = path
                return

        checkpoints = sorted(self.model_dir.glob("*.pt")) + sorted(self.model_dir.glob("*.pth"))
        if checkpoints:
            self.model_path = checkpoints[0]
        else:
            self.model_path = None

    def _extract_state_dict(self, payload: object) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(payload, dict):
            for key in [
                "state_dict",
                "generator",
                "G_state_dict",
                "netG",
                "model_state_dict",
                "G",
                "model",
            ]:
                nested = payload.get(key) if isinstance(payload, dict) else None
                if isinstance(nested, dict):
                    return _normalise_state_dict(nested)
            return _normalise_state_dict(payload)
        if isinstance(payload, nn.Module):
            return _normalise_state_dict(payload.state_dict())
        return None

    def _load(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            self.status = _BackendStatus(is_ready=False, error="Missing Face Aging GAN checkpoint")
            return

        try:
            try:
                payload = torch.load(str(self.model_path), map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(str(self.model_path), map_location="cpu")
        except Exception as load_error:
            self.status = _BackendStatus(is_ready=False, error=f"Unable to read checkpoint: {load_error}")
            return

        state_dict = self._extract_state_dict(payload)
        if not state_dict:
            self.status = _BackendStatus(
                is_ready=False,
                error=f"Checkpoint format not recognised ({type(payload).__name__})",
            )
            return

        weight_key = _find_first_weight_key(state_dict, ("0.weight", "1.weight", "model.0.weight"))
        if weight_key is None:
            self.status = _BackendStatus(is_ready=False, error="Unable to infer generator input shape")
            return

        first_weight = state_dict[weight_key]
        if first_weight.ndim != 4:
            self.status = _BackendStatus(is_ready=False, error="Unexpected generator weight dimensions")
            return

        input_channels = first_weight.shape[1]
        self.label_channels = max(0, input_channels - 3)

        generator = FaceAgingGenerator(input_channels=input_channels)
        missing, unexpected = generator.load_state_dict(state_dict, strict=False)
        is_ready = len(missing) == 0 and len(unexpected) == 0
        self.status = _BackendStatus(
            is_ready=is_ready,
            error=None if is_ready else f"Missing keys: {len(missing)}, unexpected: {len(unexpected)}",
        )

        self.model = generator.to(self.device)
        self.model.eval()

    def _encode_condition(self, current_age: float, target_age: float, batch: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.label_channels <= 0:
            return None

        current_norm = max(0.0, min(current_age, 100.0)) / 100.0
        target_norm = max(0.0, min(target_age, 100.0)) / 100.0

        if self.label_channels == 1:
            condition = torch.full((batch, 1), target_norm, dtype=dtype, device=device)
        elif self.label_channels == 2:
            condition = torch.tensor([[current_norm, target_norm]], dtype=dtype, device=device)
            if batch > 1:
                condition = condition.expand(batch, -1)
        elif self.label_channels % 2 == 0 and self.label_channels <= 20:
            half = self.label_channels // 2
            current_bins = torch.zeros((batch, half), dtype=dtype, device=device)
            target_bins = torch.zeros((batch, half), dtype=dtype, device=device)

            def _age_to_index(value: float) -> int:
                scaled = max(0.0, min(value, 0.9999)) * half
                return min(int(round(scaled)), half - 1)

            current_idx = _age_to_index(current_norm)
            target_idx = _age_to_index(target_norm)
            current_bins[:, current_idx] = 1.0
            target_bins[:, target_idx] = 1.0
            condition = torch.cat([current_bins, target_bins], dim=1)
        else:
            condition = torch.zeros((batch, self.label_channels), dtype=dtype, device=device)
            idx = min(int(round(target_norm * (self.label_channels - 1))), self.label_channels - 1)
            condition[:, idx] = 1.0

        return condition

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

            condition = self._encode_condition(current_age, target_age, batched.size(0), batched.device, batched.dtype)
            input_tensor = batched * 2.0 - 1.0
            output = self.model(input_tensor, condition)

            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.dim() == 4:
                output = output[0]

            result = torch.clamp((output + 1.0) * 0.5, 0.0, 1.0)
            return result
        except Exception as runtime_error:  # pragma: no cover - best effort fallback
            self.status = _BackendStatus(is_ready=False, error=str(runtime_error))
            return None
