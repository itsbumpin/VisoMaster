from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
from types import SimpleNamespace
import types


class _DummyModule:
    def to(self, _device):
        return self

    def eval(self):
        return self


class _DummyIdentity(_DummyModule):
    pass


dummy_nn = types.SimpleNamespace(Module=_DummyModule, Identity=_DummyIdentity)

dummy_torch = types.SimpleNamespace(
    device=lambda value: value,
    nn=dummy_nn,
    jit=types.SimpleNamespace(load=lambda *args, **kwargs: _DummyModule()),
    load=lambda *args, **kwargs: {},
)

sys.modules.setdefault("torch", dummy_torch)
sys.modules.setdefault("torch.nn", dummy_nn)
sys.modules.setdefault("torch.jit", dummy_torch.jit)

repo_root = Path(__file__).resolve().parents[3]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from app.processors.face_reaging.backend import FaceReagingBackend, _BackendStatus


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
