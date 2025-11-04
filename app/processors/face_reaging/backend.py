from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

import torch.nn as nn

from app.processors.models_data import models_dir, third_party_dir


def _default_checkpoint_candidates() -> Tuple[Path, ...]:
    """Return preferred checkpoint locations in priority order."""

    third_party_root = Path(third_party_dir)
    models_root = Path(models_dir)

    return (
        third_party_root / "SAM" / "pretrained" / "sam_ffhq_aging.pt",
        models_root / "face_reaging" / "sam_ffhq_aging.pt",
    )


@dataclass
class _BackendStatus:
    is_ready: bool = False
    error: Optional[str] = None


class FaceReagingBackend:
    """Wrapper that loads the SAM re-aging generator checkpoint if available."""

    def __init__(self, device: str | torch.device, model_path: Path | None = None) -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.model_path, self._searched_paths = self._resolve_model_path(model_path)
        for candidate in self._searched_paths:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        self.model: Optional[nn.Module] = None
        self.status = _BackendStatus(is_ready=False, error=None)
        self.condition_channels = 2
        self._load()

    @property
    def is_ready(self) -> bool:
        return self.status.is_ready and self.model is not None

    def _resolve_model_path(
        self, override: Path | None
    ) -> Tuple[Path, Tuple[Path, ...]]:
        if override is not None:
            path = Path(override)
            return path, (path,)

        candidates = _default_checkpoint_candidates()
        for candidate in candidates:
            if candidate.exists():
                return candidate, candidates

        return candidates[0], candidates

    def _load(self) -> None:
        if not self.model_path.exists():
            joined_candidates = "\n".join(f"- {path}" for path in self._searched_paths)
            error = (
                "Missing checkpoint for the SAM re-aging backend.\n"
                "Download `sam_ffhq_aging.pt` from the official SAM repository "
                "(yuval-alaluf/SAM) and place it in the cloned project under "
                "`third_party/SAM/pretrained/`.\n"
                "The backend searched the following locations:\n"
                f"{joined_candidates}"
            )
            self.status = _BackendStatus(
                is_ready=False,
                error=error,
            )
            return

        try:
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
            self.model.eval()
            self.status = _BackendStatus(is_ready=True, error=None)
            return
        except Exception:  # pragma: no cover - fallback path
            pass

        try:
            state = torch.load(str(self.model_path), map_location=self.device)
        except Exception as load_error:
            self.status = _BackendStatus(is_ready=False, error=f"Unable to read SAM checkpoint: {load_error}")
            return

        if self._materialise_model_from_checkpoint(state):
            return

        self.status = _BackendStatus(
            is_ready=False,
            error=(
                "Unsupported checkpoint format. The SAM re-aging checkpoint must be a TorchScript "
                "module, a pickled nn.Module, or a dictionary that includes the model weights "
                "and options produced by the official SAM repository."
            ),
        )

    def _materialise_model_from_checkpoint(self, checkpoint: object) -> bool:
        """Attempt to build a runnable module from various SAM checkpoint layouts."""

        if isinstance(checkpoint, nn.Module):
            self.model = checkpoint.to(self.device)
            self.model.eval()
            self.status = _BackendStatus(is_ready=True, error=None)
            return True

        if isinstance(checkpoint, dict):
            module_keys = ("model", "module", "generator", "sam", "net")
            for key in module_keys:
                candidate = checkpoint.get(key)
                if isinstance(candidate, nn.Module):
                    self.model = candidate.to(self.device)
                    self.model.eval()
                    self.status = _BackendStatus(is_ready=True, error=None)
                    return True

            state_dict_keys = (
                "state_dict",
                "model_state_dict",
                "generator",
                "ema",
                "model",
                "g",
            )
            opts_keys = ("opts", "opt", "options")

            state_dict = None
            for key in state_dict_keys:
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    state_dict = value
                    break

            opts = None
            for key in opts_keys:
                value = checkpoint.get(key)
                if value is not None:
                    opts = value
                    break

            if state_dict is not None and opts is not None:
                module = self._load_psp_from_state_dict(state_dict, opts)
                if module is not None:
                    self.model = module
                    self.model.eval()
                    self.status = _BackendStatus(is_ready=True, error=None)
                    return True

        return False

    def _load_psp_from_state_dict(
        self,
        state_dict: dict,
        opts: object,
    ) -> Optional[nn.Module]:
        try:
            from app.third_party.sam import SAMModule

            opts_dict = self._normalise_opts(opts)
            opts_dict.setdefault("checkpoint_path", str(self.model_path))
            module = SAMModule(opts_dict, state_dict, device=self.device)
            return module.to(self.device)
        except Exception as build_error:
            self.status = _BackendStatus(
                is_ready=False,
                error=f"Failed to reconstruct SAM model from checkpoint: {build_error}",
            )
            return None

    def _normalise_opts(self, opts: object) -> dict:
        if isinstance(opts, dict):
            return dict(opts)
        if isinstance(opts, SimpleNamespace):
            return vars(opts)
        if hasattr(opts, "__dict__"):
            return dict(vars(opts))
        raise TypeError("Unsupported opts object in SAM checkpoint")

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

            output = self._invoke_external_model(batched, condition, current_age, target_age)

            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.dim() == 4:
                output = output[0]

            if not output.dtype.is_floating_point:
                output = output.to(torch.float32)
            scaled = torch.tanh(output)

            result = (scaled + 1.0) * 0.5
            return torch.clamp(result, 0.0, 1.0)
        except Exception as runtime_error:  # pragma: no cover - best effort fallback
            self.status = _BackendStatus(is_ready=False, error=str(runtime_error))
            return None

    def _invoke_external_model(
        self,
        batched: torch.Tensor,
        condition: torch.Tensor,
        current_age: float,
        target_age: float,
    ) -> torch.Tensor:
        """Attempt to call a third-party SAM module with multiple fallbacks."""

        normalized = batched * 2.0 - 1.0
        condition_map = condition.view(1, self.condition_channels, 1, 1).expand(
            batched.size(0), -1, batched.shape[-2], batched.shape[-1]
        )
        # The original SAM TorchScript module consumes a remarkably flexible
        # assortment of inputs depending on the revision that produced the
        # checkpoint.  Some versions expect the normalised RGB image only,
        # others require the age conditioning vector and a pre-expanded
        # conditioning map that matches the internal 256-channel FiLM blocks.
        #
        # To cover the common variants we eagerly materialise a number of
        # tensors that can be mixed-and-matched when probing the model's
        # signature.  These are cheap to create compared to the cost of the
        # failed forward calls that would otherwise bubble up as backend
        # initialisation errors.
        combined = torch.cat([normalized, condition_map], dim=1)
        repeated_condition_256 = condition.repeat_interleave(128, dim=1)
        condition_map_256 = repeated_condition_256.view(1, -1, 1, 1).expand(
            batched.size(0), -1, batched.shape[-2], batched.shape[-1]
        )
        repeated_condition_512 = condition.repeat_interleave(256, dim=1)
        condition_map_512 = repeated_condition_512.view(1, -1, 1, 1).expand(
            batched.size(0), -1, batched.shape[-2], batched.shape[-1]
        )

        age_tensor = torch.tensor(
            [target_age],
            dtype=batched.dtype,
            device=batched.device,
        )
        current_tensor = torch.tensor(
            [current_age],
            dtype=batched.dtype,
            device=batched.device,
        )
        delta_tensor = torch.tensor(
            [(target_age - current_age) / 100.0],
            dtype=batched.dtype,
            device=batched.device,
        )

        attempts = [
            ((normalized,), {}),
            ((normalized,), {"age": age_tensor}),
            ((normalized,), {"target_age": age_tensor}),
            ((normalized,), {"age_target": age_tensor}),
            ((normalized,), {"alpha": delta_tensor}),
            ((normalized,), {"age": delta_tensor}),
            ((normalized,), {"target": delta_tensor}),
            ((normalized,), {"source_age": current_tensor, "target_age": age_tensor}),
            ((normalized,), {"input_age": current_tensor, "output_age": age_tensor}),
            ((normalized,), {"condition": condition}),
            ((normalized,), {"age_condition": condition}),
            ((normalized,), {"age_condition": condition_map}),
            ((normalized,), {"age_condition": condition_map_256}),
            ((normalized,), {"age_condition": condition_map_512}),
            ((normalized,), {"condition_map": condition_map}),
            ((normalized,), {"condition_map": condition_map_256}),
            ((normalized,), {"condition_map": condition_map_512}),
            ((normalized, condition), {}),
            ((normalized, condition_map), {}),
            ((normalized, condition_map_256), {}),
            ((normalized, condition_map_512), {}),
            ((normalized, condition, age_tensor), {}),
            ((normalized, condition_map, age_tensor), {}),
            ((normalized, condition_map_256, age_tensor), {}),
            ((normalized, condition_map_512, age_tensor), {}),
            ((normalized, current_tensor, age_tensor), {}),
            ((normalized, current_tensor, age_tensor, delta_tensor), {}),
            ((combined,), {}),
            ((combined,), {"age": age_tensor}),
            ((combined,), {"alpha": delta_tensor}),
            ((combined,), {"age_condition": condition}),
            ((combined,), {"age_condition": condition_map_256}),
            ((combined,), {"age_condition": condition_map_512}),
        ]

        last_error: Optional[Exception] = None
        forward = self.model.forward if hasattr(self.model, "forward") else self.model
        signature = None
        allows_var_kwargs = False
        try:
            signature = inspect.signature(forward)
            allows_var_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):  # pragma: no cover - dynamic modules may not expose signature
            signature = None

        for args, kwargs in attempts:
            attempt_kwargs = dict(kwargs)
            if signature is not None:
                bound_kwargs = {}
                for key, value in attempt_kwargs.items():
                    if key in signature.parameters:
                        bound_kwargs[key] = value
                if bound_kwargs:
                    attempt_kwargs = bound_kwargs
                elif not allows_var_kwargs:
                    attempt_kwargs = {}
            try:
                return self.model(*args, **attempt_kwargs)
            except TypeError as error:
                last_error = error
                continue
            except RuntimeError as error:
                last_error = error
                continue

        if last_error is None:
            last_error = RuntimeError("Unable to execute SAM model with the provided inputs")
        raise last_error
