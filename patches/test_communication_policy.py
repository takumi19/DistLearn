"""Standalone tests for communication_policy (run: python3 test_communication_policy.py).

No torch / no full package needed — imports the module by file path so it can be
verified locally before deploying to the cluster. Also runnable under pytest.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "communication_policy", os.path.join(_HERE, "communication_policy.py")
)
cp = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve string annotations (PEP 563).
sys.modules["communication_policy"] = cp
_spec.loader.exec_module(cp)


@dataclass
class FakeNeighbor:
    id: str
    host: str = "10.0.0.1"
    port: int = 50051


def _neighbors(ids):
    return [FakeNeighbor(id=i) for i in ids]


def _ids(selected):
    return [n.id for n in selected]


# ── Backward compatibility: full policy == original rotation ──────────────────

def test_full_returns_all_when_fanout_zero():
    ns = _neighbors(["c", "a", "b"])
    out = cp.select_push_neighbors(ns, policy="full", push_fanout=0)
    assert _ids(out) == ["a", "b", "c"], "full+fanout0 must return all, sorted"


def test_full_rotation_matches_original_formula():
    ns = _neighbors(["a", "b", "c", "d", "e"])
    step, node = 7, "decentr-03"
    out = cp.select_push_neighbors(
        ns, policy="full", push_fanout=2, current_step=step, self_node_id=node
    )
    ordered = sorted(["a", "b", "c", "d", "e"])
    offset = (step + sum(ord(ch) for ch in node)) % len(ordered)
    expected = (ordered[offset:] + ordered[:offset])[:2]
    assert _ids(out) == expected, f"rotation drift: {_ids(out)} != {expected}"


def test_random_fanout_is_alias_of_full():
    ns = _neighbors(["a", "b", "c", "d"])
    a = cp.select_push_neighbors(ns, policy="full", push_fanout=2, current_step=3, self_node_id="x")
    b = cp.select_push_neighbors(ns, policy="random_fanout", push_fanout=2, current_step=3, self_node_id="x")
    assert _ids(a) == _ids(b)


# ── Score-based policies ──────────────────────────────────────────────────────

def test_top_k_fastest_picks_highest_capacity():
    ns = _neighbors(["a", "b", "c"])
    t = cp.PeerScoreTracker()
    t.seed_capacity("a", 0.3)
    t.seed_capacity("b", 2.0)   # fastest
    t.seed_capacity("c", 1.0)
    out = cp.select_push_neighbors(ns, policy="top_k_fastest", push_fanout=2, tracker=t)
    assert _ids(out)[0] == "b", "fastest peer must rank first"
    assert len(out) == 2


def test_top_k_reliable_prefers_high_success():
    ns = _neighbors(["a", "b"])
    t = cp.PeerScoreTracker(success_alpha=0.5)
    for _ in range(5):
        t.record_push("a", ok=True, latency_s=0.05)
        t.record_push("b", ok=False, latency_s=0.05)
    out = cp.select_push_neighbors(ns, policy="top_k_reliable", push_fanout=1, tracker=t)
    assert _ids(out) == ["a"], "reliable peer must win"


def test_ping_aware_penalises_latency():
    ns = _neighbors(["near", "far"])
    t = cp.PeerScoreTracker(latency_alpha=1.0, success_alpha=1.0)
    t.seed_capacity("near", 1.0)
    t.seed_capacity("far", 1.0)
    t.record_push("near", ok=True, latency_s=0.01)   # 10 ms
    t.record_push("far", ok=True, latency_s=0.50)    # 500 ms
    out = cp.select_push_neighbors(ns, policy="ping_aware", push_fanout=1, tracker=t)
    assert _ids(out) == ["near"], "low-latency peer must win ping_aware"


def test_top_k_useful_falls_back_to_reliability_without_data():
    ns = _neighbors(["a", "b"])
    t = cp.PeerScoreTracker(success_alpha=1.0)
    t.record_push("a", ok=True, latency_s=0.05)
    t.record_push("b", ok=False, latency_s=0.05)
    out = cp.select_push_neighbors(ns, policy="top_k_useful", push_fanout=1, tracker=t)
    assert _ids(out) == ["a"], "without usefulness data, fall back to success"


def test_top_k_useful_uses_usefulness_when_present():
    ns = _neighbors(["a", "b"])
    t = cp.PeerScoreTracker(usefulness_alpha=1.0, success_alpha=1.0)
    # a is reliable but useless; b is less reliable but very useful
    t.record_push("a", ok=True, latency_s=0.05)
    t.record_push("b", ok=True, latency_s=0.05)
    t.record_usefulness("a", 0.01)
    t.record_usefulness("b", 0.50)
    out = cp.select_push_neighbors(ns, policy="top_k_useful", push_fanout=1, tracker=t)
    assert _ids(out) == ["b"], "useful peer must win once usefulness is known"


# ── reliability_aware graph pruning ───────────────────────────────────────────

def test_reliability_aware_prunes_bad_edges():
    ns = _neighbors(["good", "bad"])
    t = cp.PeerScoreTracker(success_alpha=0.6)
    for _ in range(8):
        t.record_push("good", ok=True, latency_s=0.05)
        t.record_push("bad", ok=False, latency_s=0.05)
    out = cp.select_push_neighbors(ns, policy="reliability_aware", tracker=t)
    assert _ids(out) == ["good"], "bad edge must be pruned"


def test_reliability_aware_never_isolates():
    ns = _neighbors(["x", "y", "z"])
    t = cp.PeerScoreTracker(success_alpha=0.9)
    for _ in range(10):  # everyone fails — must still keep the best one
        for nid in ("x", "y", "z"):
            t.record_push(nid, ok=False, latency_s=0.05)
    out = cp.select_push_neighbors(ns, policy="reliability_aware", tracker=t)
    assert len(out) >= 1, "must never isolate a node"


# ── Safety / validation ───────────────────────────────────────────────────────

def test_unknown_policy_raises():
    ns = _neighbors(["a"])
    try:
        cp.select_push_neighbors(ns, policy="bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("unknown policy must raise ValueError")


def test_score_policy_without_tracker_falls_back():
    ns = _neighbors(["a", "b", "c"])
    out = cp.select_push_neighbors(ns, policy="top_k_fastest", push_fanout=2, tracker=None)
    assert len(out) == 2, "no tracker -> safe rotation fallback, no crash"


def test_empty_neighbors():
    assert cp.select_push_neighbors([], policy="top_k_fastest", tracker=cp.PeerScoreTracker()) == []


def test_tracker_success_rate_moves():
    t = cp.PeerScoreTracker(success_alpha=0.5)
    t.record_push("p", ok=False, latency_s=0.1)
    assert t.success_rate("p") < 1.0
    for _ in range(20):
        t.record_push("p", ok=True, latency_s=0.1)
    assert t.success_rate("p") > 0.9


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
