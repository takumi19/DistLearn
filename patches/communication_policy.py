"""Peer-selection policies for async gossip push targets.

Deploys to: decentr_my_own/algorithms/communication_policy.py

This module decides WHICH neighbours a node pushes its weights to on each gossip
step. It is intentionally dependency-free (pure Python, no torch) so the
selection logic can be unit-tested in isolation and reused by tests/tools.

Backward compatibility
----------------------
``policy="full"`` reproduces the original ``_select_push_neighbors`` behaviour
byte-for-byte (sorted by id; if ``push_fanout`` is in ``(0, len)`` rotate by a
step+node-id offset and take the first ``fanout``). Every experiment that does
not set ``async.communication_policy`` therefore behaves exactly as before.

Scoring signals
---------------
All scores use signals already available in the training loop — no extra RPCs:
  * capacity        — neighbour ``resources.relative_speed`` (seeded once)
  * success_rate    — EMA of push ok/fail outcomes
  * latency         — EMA of measured push round-trip seconds
  * usefulness      — EMA of training-loss reduction after mixing (proxy; the
                      roadmap flags an honest usefulness signal as hard, so this
                      is a documented approximation, derived from train/val only,
                      never the test set)
"""
from __future__ import annotations

from dataclasses import dataclass

# Policies that rank neighbours by a peer score (need a PeerScoreTracker).
_SCORE_POLICIES = frozenset(
    {"top_k_fastest", "top_k_reliable", "ping_aware", "top_k_useful"}
)
# All recognised policies. Unknown policy -> ValueError (fail fast at config time).
VALID_POLICIES = _SCORE_POLICIES | {"full", "random_fanout", "reliability_aware", "dynamic"}

# reliability_aware: drop static-graph edges whose success EMA is below this,
# but never isolate a node — always keep at least this many best neighbours.
_RELIABILITY_MIN_SUCCESS = 0.5
_RELIABILITY_MIN_DEGREE = 1


def policy_uses_scores(policy: str) -> bool:
    """True if the policy ranks neighbours via PeerScoreTracker scores."""
    return policy in _SCORE_POLICIES


@dataclass
class _PeerStats:
    # success/usefulness start optimistic so an unseen peer is tried at least
    # once before the tracker has evidence to deprioritise it.
    capacity: float = 1.0
    success_ema: float = 1.0
    latency_ema_s: float = 0.05
    usefulness_ema: float = 0.0
    has_usefulness: bool = False
    push_count: int = 0
    ok_count: int = 0


class PeerScoreTracker:
    """Rolling per-peer scores updated during a run.

    One instance lives for the whole run and is consulted by
    :func:`select_push_neighbors` for score-based policies. Updates are cheap
    EMAs so the tracker reacts to changing WAN conditions without oscillating.
    """

    def __init__(
        self,
        *,
        success_alpha: float = 0.3,
        latency_alpha: float = 0.3,
        usefulness_alpha: float = 0.3,
    ) -> None:
        self._success_alpha = float(success_alpha)
        self._latency_alpha = float(latency_alpha)
        self._usefulness_alpha = float(usefulness_alpha)
        self._peers: dict[str, _PeerStats] = {}

    def _stats(self, peer_id: str) -> _PeerStats:
        stats = self._peers.get(peer_id)
        if stats is None:
            stats = _PeerStats()
            self._peers[peer_id] = stats
        return stats

    def seed_capacity(self, peer_id: str, relative_speed: float) -> None:
        """Set a peer's compute capacity (neighbour relative_speed)."""
        self._stats(peer_id).capacity = max(float(relative_speed), 1e-6)

    def record_push(self, peer_id: str, *, ok: bool, latency_s: float) -> None:
        """Fold one push outcome into the peer's success/latency EMAs."""
        stats = self._stats(peer_id)
        stats.push_count += 1
        if ok:
            stats.ok_count += 1
        observed = 1.0 if ok else 0.0
        stats.success_ema = (
            self._success_alpha * observed
            + (1.0 - self._success_alpha) * stats.success_ema
        )
        # Only successful pushes carry a meaningful latency sample.
        if ok and latency_s > 0:
            stats.latency_ema_s = (
                self._latency_alpha * float(latency_s)
                + (1.0 - self._latency_alpha) * stats.latency_ema_s
            )

    def record_usefulness(self, peer_id: str, val_delta: float) -> None:
        """Fold a usefulness sample (e.g. loss reduction after mixing) into EMA."""
        stats = self._stats(peer_id)
        if stats.has_usefulness:
            stats.usefulness_ema = (
                self._usefulness_alpha * float(val_delta)
                + (1.0 - self._usefulness_alpha) * stats.usefulness_ema
            )
        else:
            stats.usefulness_ema = float(val_delta)
            stats.has_usefulness = True

    def success_rate(self, peer_id: str) -> float:
        return self._stats(peer_id).success_ema

    def score(self, peer_id: str, policy: str) -> float:
        """Score a peer under the given policy (higher = prefer to push)."""
        s = self._stats(peer_id)
        if policy == "top_k_fastest":
            # Prefer fast peers, lightly penalised by latency so a fast-but-
            # unreachable peer doesn't dominate.
            return s.capacity / (1.0 + s.latency_ema_s)
        if policy == "top_k_reliable":
            return s.success_ema / (1.0 + s.latency_ema_s)
        if policy == "ping_aware":
            latency_ms = s.latency_ema_s * 1000.0
            return s.success_ema * s.capacity / (1.0 + latency_ms)
        if policy == "top_k_useful":
            # Fall back to reliability until usefulness evidence exists.
            if not s.has_usefulness:
                return s.success_ema
            return s.usefulness_ema
        # Non-score policies never reach here.
        return 0.0

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Plain-dict view of all peer stats for metrics/debugging."""
        return {
            pid: {
                "capacity": st.capacity,
                "success_ema": st.success_ema,
                "latency_ema_s": st.latency_ema_s,
                "usefulness_ema": st.usefulness_ema,
                "push_count": st.push_count,
                "ok_count": st.ok_count,
            }
            for pid, st in self._peers.items()
        }


def _rotate_fanout(ordered: list, *, push_fanout: int, current_step: int, self_node_id: str) -> list:
    """Original rotation behaviour (policy='full'/'random_fanout')."""
    if push_fanout <= 0 or push_fanout >= len(ordered):
        return ordered
    offset_seed = current_step + sum(ord(char) for char in self_node_id)
    offset = offset_seed % len(ordered)
    rotated = ordered[offset:] + ordered[:offset]
    return rotated[:push_fanout]


def select_push_neighbors(
    neighbors: list,
    *,
    policy: str = "full",
    push_fanout: int = 0,
    current_step: int = 0,
    self_node_id: str = "",
    tracker: PeerScoreTracker | None = None,
) -> list:
    """Return the subset of ``neighbors`` to push model weights to this step.

    Parameters mirror the call site in async_gossip. ``neighbors`` are objects
    with ``.id`` (and ``.host``/``.port`` used downstream). Selection is stable
    and deterministic given the same inputs so behaviour is reproducible.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(
            f"Unknown communication_policy '{policy}'. "
            f"Valid: {', '.join(sorted(VALID_POLICIES))}"
        )

    ordered = sorted(neighbors, key=lambda item: item.id)
    if not ordered:
        return ordered

    # ── full / random_fanout: original rotation (backward compatible) ─────────
    if policy in ("full", "random_fanout"):
        return _rotate_fanout(
            ordered,
            push_fanout=push_fanout,
            current_step=current_step,
            self_node_id=self_node_id,
        )

    # ── reliability_aware: prune low-success edges on the static graph ────────
    if policy == "reliability_aware":
        if tracker is None:
            return ordered
        good = [n for n in ordered if tracker.success_rate(n.id) >= _RELIABILITY_MIN_SUCCESS]
        if len(good) >= _RELIABILITY_MIN_DEGREE:
            return good
        # All edges look bad — keep the best few so the node is never isolated.
        ranked = sorted(ordered, key=lambda n: (-tracker.success_rate(n.id), n.id))
        return ranked[:_RELIABILITY_MIN_DEGREE]

    # ── score-based top-k policies ────────────────────────────────────────────
    if tracker is None:
        # No scores yet (shouldn't happen in the run loop) — safe fallback.
        return _rotate_fanout(
            ordered,
            push_fanout=push_fanout,
            current_step=current_step,
            self_node_id=self_node_id,
        )
    ranked = sorted(ordered, key=lambda n: (-tracker.score(n.id, policy), n.id))
    if push_fanout <= 0 or push_fanout >= len(ranked):
        return ranked
    return ranked[:push_fanout]
