"""Standalone tests for Phase-3 helpers in async_gossip.py.

Tests the pure tensor helpers (_compress_state, _decompress_payload) and the
DeltaExchangeState dataclass without standing up the full training stack: the
heavy ``decentr_my_own.*`` imports are stubbed in sys.modules so async_gossip
can be imported by file path.

Run: python3 test_phase3_compression.py   (or under pytest)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _install_stubs() -> None:
    """Register dummy decentr_my_own.* modules so async_gossip imports cleanly."""
    class _Any:
        def __init__(self, *a, **k): ...
        def __call__(self, *a, **k): return None

    def _stub(name: str, **attrs) -> None:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        # Anything not explicitly set resolves to a no-op class.
        mod.__getattr__ = lambda n: _Any  # type: ignore[attr-defined]
        sys.modules[name] = mod

    # The real communication_policy module loads fine (pure python) — but to keep
    # this test hermetic we point the package import at our local file.
    cp_spec = importlib.util.spec_from_file_location(
        "decentr_my_own.algorithms.communication_policy",
        os.path.join(_HERE, "communication_policy.py"),
    )
    cp = importlib.util.module_from_spec(cp_spec)
    sys.modules["decentr_my_own.algorithms.communication_policy"] = cp
    cp_spec.loader.exec_module(cp)

    for name in [
        "decentr_my_own", "decentr_my_own.algorithms",
        "decentr_my_own.comm", "decentr_my_own.comm.client",
        "decentr_my_own.comm.messages", "decentr_my_own.comm.server",
        "decentr_my_own.config", "decentr_my_own.config.loader",
        "decentr_my_own.config.models", "decentr_my_own.data",
        "decentr_my_own.data.runtime", "decentr_my_own.data.loaders",
        "decentr_my_own.data.manifest", "decentr_my_own.data.scheduler_state",
        "decentr_my_own.data.shards", "decentr_my_own.models",
        "decentr_my_own.models.factory", "decentr_my_own.training",
        "decentr_my_own.training.device", "decentr_my_own.training.io",
        "decentr_my_own.training.metrics", "decentr_my_own.training.seed",
        "decentr_my_own.training.state_ops",
    ]:
        if name not in sys.modules:
            _stub(name)


def _load_async_gossip():
    _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "async_gossip_under_test", os.path.join(_HERE, "async_gossip.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["async_gossip_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


ag = _load_async_gossip()


def _sample_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "layer.weight": torch.randn(8, 4),
        "layer.bias": torch.randn(8),
        # non-float buffer must pass through compression untouched
        "bn.num_batches_tracked": torch.tensor(7, dtype=torch.long),
    }


def _max_abs_diff(a: dict, b: dict) -> float:
    return max(
        float((a[k].float() - b[k].float()).abs().max())
        for k in a
        if torch.is_floating_point(a[k])
    )


# ── compression round-trips ───────────────────────────────────────────────────

def test_none_is_identity():
    state = _sample_state()
    tensors, kind = ag._compress_state(state, "none", 0.1)
    assert kind == "async_weights"
    assert tensors is state  # no-op returns the same dict
    back = ag._decompress_payload(tensors, kind, state)
    assert _max_abs_diff(state, back) == 0.0


def test_float16_roundtrip_close():
    state = _sample_state()
    tensors, kind = ag._compress_state(state, "float16", 0.1)
    assert kind == "async_weights_f16"
    assert tensors["layer.weight"].dtype == torch.float16
    # non-float buffer untouched
    assert tensors["bn.num_batches_tracked"].dtype == torch.long
    back = ag._decompress_payload(tensors, kind, state)
    assert back["layer.weight"].dtype == torch.float32
    assert _max_abs_diff(state, back) < 1e-2


def test_quant8_roundtrip_bounded_error():
    state = _sample_state()
    tensors, kind = ag._compress_state(state, "quant8", 0.1)
    assert kind == "async_weights_q8"
    assert tensors["layer.weight"].dtype == torch.uint8
    assert "__q8s__layer.weight" in tensors  # scale metadata present
    back = ag._decompress_payload(tensors, kind, state)
    # 8-bit over the tensor range → error bounded by range/255.
    w = state["layer.weight"]
    rng = float(w.max() - w.min())
    assert _max_abs_diff(state, back) <= rng / 255.0 + 1e-5
    # scale keys must not leak into the reconstructed state
    assert set(back) == set(state)


def test_topk_sparse_keeps_largest():
    state = {"w": torch.tensor([0.1, -5.0, 0.2, 4.0, -0.3, 0.05])}
    tensors, kind = ag._compress_state(state, "topk_sparse", 1 / 3)  # keep 2 of 6
    assert kind == "async_weights"
    back = ag._decompress_payload(tensors, kind, state)
    nonzero = (back["w"].abs() > 0).sum().item()
    assert nonzero == 2
    # the two kept entries are the largest-magnitude ones (-5.0, 4.0)
    assert back["w"][1].item() != 0.0
    assert back["w"][3].item() != 0.0


def test_decompress_unknown_kind_returns_none():
    state = _sample_state()
    assert ag._decompress_payload(state, "totally_unknown", state) is None


def test_quant8_missing_scale_returns_none():
    state = _sample_state()
    tensors, kind = ag._compress_state(state, "quant8", 0.1)
    del tensors["__q8s__layer.weight"]  # corrupt the payload
    assert ag._decompress_payload(tensors, kind, state) is None


# ── DeltaExchangeState ────────────────────────────────────────────────────────

def test_delta_state_defaults_independent():
    a = ag.DeltaExchangeState()
    b = ag.DeltaExchangeState()
    assert a.last_sent is None
    a.reconstructions["x"] = 1
    assert b.reconstructions == {}  # __post_init__ gives each its own dict


def test_delta_reconstruction_math():
    # Simulate: base state + delta == new state (the receiver's reconstruction).
    base = {"w": torch.tensor([1.0, 2.0, 3.0])}
    new = {"w": torch.tensor([1.5, 2.5, 2.0])}
    delta = {"w": new["w"] - base["w"]}
    reconstructed = {"w": base["w"] + delta["w"]}
    assert torch.allclose(reconstructed["w"], new["w"])


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
