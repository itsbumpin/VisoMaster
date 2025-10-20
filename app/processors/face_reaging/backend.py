from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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


class _FaceAgingGanBackend:
    """Wrapper that loads the Face Aging and De-aging GAN generator."""

    label = "Face Aging GAN"

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


class _SAMBackend:
    """Wrapper that loads a SAM (StyleGAN-based) de-aging checkpoint when available."""

    label = "SAM De-Aging"

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model_dir = Path(models_dir) / "face_reaging"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path: Optional[Path] = None
        self.model: Optional[torch.nn.Module] = None
        self.status = _BackendStatus(is_ready=False, error="SAM backend not initialised")
        self._resolve_checkpoint()
        self._load()

    @property
    def is_ready(self) -> bool:
        return self.status.is_ready and self.model is not None

    def _resolve_checkpoint(self) -> None:
        candidates = [
            "sam_deaging.pt",
            "sam_deaging.pth",
            "SAM_ffhq_deaging.pt",
            "SAM_ffhq_deaging.pth",
            "sam_deaging.ts",
            "sam_deaging.jit",
        ]
        for candidate in candidates:
            path = self.model_dir / candidate
            if path.exists():
                self.model_path = path
                return

        sam_dir = self.model_dir / "sam"
        checkpoints: list[Path] = []
        if sam_dir.exists():
            checkpoints.extend(sorted(sam_dir.glob("**/*.pt")))
            checkpoints.extend(sorted(sam_dir.glob("**/*.pth")))
            checkpoints.extend(sorted(sam_dir.glob("**/*.ts")))
            checkpoints.extend(sorted(sam_dir.glob("**/*.jit")))

        if checkpoints:
            self.model_path = checkpoints[0]
        else:
            self.model_path = None

    def _try_load_script(self) -> bool:
        if self.model_path is None:
            return False
        try:
            scripted = torch.jit.load(str(self.model_path), map_location=self.device)
        except Exception:
            return False
        self.model = scripted.to(self.device)
        self.model.eval()
        self.status = _BackendStatus(is_ready=True, error=None)
        return True

    def _extract_module_from_payload(self, payload: object) -> Optional[nn.Module]:
        if isinstance(payload, (nn.Module, torch.jit.ScriptModule)):
            return payload  # type: ignore[return-value]
        if isinstance(payload, dict):
            for key in [
                "model",
                "generator",
                "sam",
                "module",
                "network",
            ]:
                nested = payload.get(key)
                if isinstance(nested, (nn.Module, torch.jit.ScriptModule)):
                    return nested  # type: ignore[return-value]
        return None

    def _load(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            self.status = _BackendStatus(is_ready=False, error="Missing SAM checkpoint")
            return

        if self._try_load_script():
            return

        try:
            payload = torch.load(str(self.model_path), map_location="cpu")
        except Exception as load_error:
            self.status = _BackendStatus(is_ready=False, error=f"Unable to read SAM checkpoint: {load_error}")
            return

        module = self._extract_module_from_payload(payload)
        if module is None:
            self.status = _BackendStatus(
                is_ready=False,
                error=f"SAM checkpoint format not recognised ({type(payload).__name__})",
            )
            return

        try:
            module = module.to(self.device)
        except Exception:
            # Some pickled modules may not implement .to; keep them on CPU but mark unusable.
            self.status = _BackendStatus(is_ready=False, error="SAM module does not support device transfer")
            return

        module.eval()
        self.model = module
        self.status = _BackendStatus(is_ready=True, error=None)

    def _build_conditions(
        self,
        batch_size: int,
        current_age: float,
        target_age: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        current_clamped = max(0.0, min(current_age, 100.0))
        target_clamped = max(0.0, min(target_age, 100.0))

        current_norm = torch.tensor([current_clamped / 100.0], dtype=dtype, device=device)
        target_norm = torch.tensor([target_clamped / 100.0], dtype=dtype, device=device)
        if batch_size > 1:
            current_norm = current_norm.expand(batch_size)
            target_norm = target_norm.expand(batch_size)
        return current_norm, target_norm, current_clamped, target_clamped

    def _extract_image(self, output: Any) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for key in (
                'image',
                'images',
                'result',
                'results',
                'output',
                'outputs',
                'y_hat',
            ):
                if key in output:
                    candidate = self._extract_image(output[key])
                    if candidate is not None:
                        return candidate
        if isinstance(output, (list, tuple)):
            for item in output:
                candidate = self._extract_image(item)
                if candidate is not None:
                    return candidate
        return None

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
            current_cond, target_cond, current_value, target_value = self._build_conditions(
                batched.size(0), current_age, target_age, batched.device, batched.dtype
            )
            normalized = batched * 2.0 - 1.0

            target_scalar = float(target_value)
            current_scalar = float(current_value)

            attempts = [
                ((normalized,), {}),
                ((normalized,), {"target_age": target_scalar}),
                ((normalized,), {"target": target_scalar}),
                ((normalized,), {"age": target_scalar}),
                ((normalized,), {"current_age": current_scalar, "target_age": target_scalar}),
                ((normalized,), {"source_age": current_scalar, "target_age": target_scalar}),
                ((normalized,), {"current": current_scalar, "target": target_scalar}),
                ((normalized,), {"source": current_scalar, "target": target_scalar}),
                ((normalized, target_scalar), {}),
                ((normalized, target_cond), {}),
                ((normalized, target_cond, current_cond), {}),
                ((normalized,), {"target": target_cond}),
                ((normalized,), {"target_age": target_cond}),
                ((normalized,), {"current": current_cond, "target": target_cond}),
                ((normalized,), {"current_age": current_cond, "target_age": target_cond}),
            ]

            output: Optional[torch.Tensor] = None
            last_error: Optional[Exception] = None
            with torch.no_grad():
                for args, kwargs in attempts:
                    try:
                        result = self.model(*args, **kwargs)
                        output = self._extract_image(result)
                        if output is not None:
                            break
                    except TypeError as exc:
                        last_error = exc
                        continue
                    except RuntimeError as exc:
                        last_error = exc
                        continue

            if output is None:
                message = "SAM forward signature not recognised"
                if last_error is not None:
                    message = f"{message}: {last_error}"
                self.status = _BackendStatus(is_ready=False, error=message)
                return None

            if output.dim() == 4:
                output = output[0]

            result = torch.clamp((output + 1.0) * 0.5, 0.0, 1.0)
            self.status = _BackendStatus(is_ready=True, error=None)
            return result
        except Exception as runtime_error:  # pragma: no cover - best effort fallback
            self.status = _BackendStatus(is_ready=False, error=str(runtime_error))
            return None


class FaceReagingBackend:
    """Facade that exposes multiple face re-aging backends with a unified interface."""

    _DEFAULT_KEY = "face_aging_gan"

    def __init__(self, device: str | torch.device) -> None:
        self._backends: dict[str, tuple[str, object]] = {
            self._DEFAULT_KEY: (_FaceAgingGanBackend.label, _FaceAgingGanBackend(device)),
            "sam": (_SAMBackend.label, _SAMBackend(device)),
        }

    @property
    def default_label(self) -> str:
        return self._backends[self._DEFAULT_KEY][0]

    def available_models(self) -> dict[str, str]:
        return {friendly: key for key, (friendly, _) in self._backends.items()}

    def resolve_label(self, label: Optional[str]) -> str:
        if label:
            for key, (friendly, _) in self._backends.items():
                if friendly == label:
                    return key
        return self._DEFAULT_KEY

    def status(self, key: str) -> _BackendStatus:
        backend_entry = self._backends.get(key)
        if backend_entry is None:
            return _BackendStatus(is_ready=False, error="Unknown backend")
        backend = backend_entry[1]
        backend_status = getattr(backend, "status", None)
        if isinstance(backend_status, _BackendStatus):
            return backend_status
        return _BackendStatus(is_ready=False, error="Unavailable status")

    def __call__(
        self,
        image: torch.Tensor,
        current_age: float,
        target_age: float,
        model_key: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        key = model_key or self._DEFAULT_KEY
        backend_entry = self._backends.get(key)
        if backend_entry is None:
            backend_entry = self._backends[self._DEFAULT_KEY]
            key = self._DEFAULT_KEY

        _, backend_impl = backend_entry
        result = backend_impl(image, current_age, target_age)  # type: ignore[misc]
        if result is None:
            status = self.status(key)
            if status.is_ready:
                return result
        return result
