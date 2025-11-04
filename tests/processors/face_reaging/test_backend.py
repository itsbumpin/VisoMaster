from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
from types import SimpleNamespace
import types


class _DummyModule:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, _device):
        return self

    def eval(self):
        return self


class _DummyIdentity(_DummyModule):
    pass


dummy_nn = types.SimpleNamespace(Module=_DummyModule, Identity=_DummyIdentity)
dummy_nn.functional = types.SimpleNamespace()

dummy_torch = types.SimpleNamespace(
    device=lambda value: value,
    nn=dummy_nn,
    jit=types.SimpleNamespace(load=lambda *args, **kwargs: _DummyModule()),
    load=lambda *args, **kwargs: {},
    functional=types.SimpleNamespace(),
)

sys.modules.setdefault("torch", dummy_torch)
sys.modules.setdefault("torch.nn", dummy_nn)
sys.modules.setdefault("torch.jit", dummy_torch.jit)

repo_root = Path(__file__).resolve().parents[3]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from app.processors.face_reaging.backend import FaceReagingBackend, _BackendStatus
import app.third_party.sam as sam_module


def test_materialise_model_handles_model_state_dict():
    backend = FaceReagingBackend.__new__(FaceReagingBackend)
    backend.device = "cpu"
    backend.model = None
    backend.status = _BackendStatus()
    backend._load_psp_from_state_dict = lambda state_dict, opts: dummy_nn.Identity()

    checkpoint = {
        "model": OrderedDict(),
        "opts": SimpleNamespace(),
    }

    assert backend._materialise_model_from_checkpoint(checkpoint) is True
    assert isinstance(backend.model, dummy_nn.Identity)
    assert backend.status.is_ready
    assert backend.status.error is None
    assert backend.is_ready


def test_load_psp_from_state_dict_uses_bundled_module(tmp_path):
    backend = FaceReagingBackend.__new__(FaceReagingBackend)
    backend.device = "cpu"
    backend.model = None
    backend.model_path = tmp_path / "sam_ffhq_aging.pt"
    backend.status = _BackendStatus()

    calls = {}

    class _FakeModule:
        def __init__(self, opts, state_dict, device=None):
            calls["opts"] = opts
            calls["state_dict"] = state_dict
            calls["device"] = device

        def to(self, device):
            calls["to_device"] = device
            return self

    original = sam_module.SAMModule
    sam_module.SAMModule = _FakeModule  # type: ignore[assignment]
    try:
        state_dict = {"weight": 1}
        opts = SimpleNamespace(output_size=32, n_styles=4, input_nc=3)
        result = backend._load_psp_from_state_dict(state_dict, opts)
    finally:
        sam_module.SAMModule = original  # type: ignore[assignment]

    assert result is not None
    assert calls["state_dict"] == state_dict
    assert calls["device"] == backend.device
    assert calls["to_device"] == backend.device
    assert calls["opts"]["output_size"] == 32


def test_sam_module_from_checkpoint_passes_through_state():
    captured = {}

    def fake_init(self, opts, state_dict, device=None):  # pragma: no cover - patched in test
        captured["opts"] = opts
        captured["state_dict"] = state_dict
        captured["device"] = device

    original_init = sam_module.SAMModule.__init__  # type: ignore[assignment]
    sam_module.SAMModule.__init__ = fake_init  # type: ignore[assignment]
    try:
        checkpoint = {"state_dict": {"weight": 1}, "opts": {"foo": "bar"}}
        module = sam_module.SAMModule.from_checkpoint(checkpoint, device="cpu")
    finally:
        sam_module.SAMModule.__init__ = original_init  # type: ignore[assignment]

    assert isinstance(module, sam_module.SAMModule)
    assert captured["opts"] == {"foo": "bar"}
    assert captured["state_dict"] == {"weight": 1}
    assert captured["device"] == "cpu"
