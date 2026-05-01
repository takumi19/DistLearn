from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import decentr_my_own.algorithms.async_gossip as async_gossip
from decentr_my_own.algorithms.async_gossip import (
    PushAttemptResult,
    _exchange_async_update,
    _mix_with_latest_peer_payloads,
    _push_payload_best_effort,
    run_async_smoke,
)
from decentr_my_own.comm.messages import PayloadMetadata, PeerPayload, StoredPayloadSummary


class _FakeServer:
    def __init__(self, payloads: list[PeerPayload]):
        self._payloads = {
            (payload.metadata.sender_node_id, payload.metadata.payload_id): payload
            for payload in payloads
        }
        self._summaries = [_summary_for_payload(payload) for payload in payloads]

    def snapshot(self):
        return SimpleNamespace(payloads=self._summaries)

    def get_payload(self, sender_node_id: str, payload_id: str | None = None):
        if payload_id is None:
            for (sender, _), payload in self._payloads.items():
                if sender == sender_node_id:
                    return payload
            return None
        return self._payloads.get((sender_node_id, payload_id))


def _summary_for_payload(payload: PeerPayload) -> StoredPayloadSummary:
    return StoredPayloadSummary(
        receiver_node_id="local",
        sender_node_id=payload.metadata.sender_node_id,
        payload_id=payload.metadata.payload_id,
        payload_kind=payload.metadata.payload_kind,
        model_version=payload.metadata.model_version,
        step=payload.metadata.step,
        sample_count=payload.metadata.sample_count,
        tensor_count=len(payload.tensors),
        num_bytes=0,
        digest="test",
        received_at="2026-01-01T00:00:00Z",
        tensor_names=tuple(payload.tensors),
    )


def _payload(
    *,
    sender: str = "peer",
    payload_id: str = "payload-1",
    version: int = 10,
    sample_count: int = 1,
    state: dict[str, torch.Tensor] | None = None,
) -> PeerPayload:
    return PeerPayload(
        metadata=PayloadMetadata(
            sender_node_id=sender,
            payload_id=payload_id,
            payload_kind="async_weights",
            model_version=version,
            step=version,
            sample_count=sample_count,
        ),
        tensors=state or {"weight": torch.tensor([10.0])},
    )


def _build_optimizer_with_state(model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return optimizer


class AsyncGossipMixingUnitTests(unittest.TestCase):
    def test_finite_peer_payload_is_mixed(self) -> None:
        result = _mix_with_latest_peer_payloads(
            current_state={"weight": torch.tensor([0.0])},
            current_version=10,
            local_sample_count=1,
            server=_FakeServer([_payload(version=10)]),
            neighbor_ids=["peer"],
            base_alpha=1.0,
            max_staleness=100,
            push_interval_steps=10,
            last_mixed_payload_ids={},
        )

        self.assertEqual(result["mixed_peer_updates"], 1)
        self.assertEqual(result["dropped_peer_updates"], 0)
        self.assertTrue(torch.allclose(result["state"]["weight"], torch.tensor([5.0])))

    def test_nonfinite_peer_payload_is_dropped_without_changing_state(self) -> None:
        result = _mix_with_latest_peer_payloads(
            current_state={"weight": torch.tensor([0.0])},
            current_version=10,
            local_sample_count=1,
            server=_FakeServer(
                [_payload(version=10, state={"weight": torch.tensor([float("nan")])})]
            ),
            neighbor_ids=["peer"],
            base_alpha=1.0,
            max_staleness=100,
            push_interval_steps=10,
            last_mixed_payload_ids={},
        )

        self.assertEqual(result["mixed_peer_updates"], 0)
        self.assertEqual(result["dropped_peer_updates"], 1)
        self.assertEqual(result["nonfinite_peer_updates"], 1)
        self.assertEqual(result["dropped_peer_senders"], ["peer"])
        self.assertIn("nonfinite_payload", result["drop_reasons"][0])
        self.assertTrue(torch.equal(result["state"]["weight"], torch.tensor([0.0])))

    def test_payload_with_extreme_normalized_version_gap_is_dropped(self) -> None:
        result = _mix_with_latest_peer_payloads(
            current_state={"weight": torch.tensor([0.0])},
            current_version=10000,
            local_sample_count=1,
            server=_FakeServer([_payload(version=1)]),
            neighbor_ids=["peer"],
            base_alpha=1.0,
            max_staleness=10,
            push_interval_steps=10,
            last_mixed_payload_ids={},
        )

        self.assertEqual(result["mixed_peer_updates"], 0)
        self.assertEqual(result["dropped_peer_updates"], 1)
        self.assertEqual(result["hard_dropped_stale_peer_updates"], 1)
        self.assertIn("normalized_version_gap", result["drop_reasons"][0])
        self.assertTrue(torch.equal(result["state"]["weight"], torch.tensor([0.0])))

    def test_future_payload_uses_gap_decay(self) -> None:
        result = _mix_with_latest_peer_payloads(
            current_state={"weight": torch.tensor([0.0])},
            current_version=100,
            local_sample_count=1,
            server=_FakeServer([_payload(version=110)]),
            neighbor_ids=["peer"],
            base_alpha=1.0,
            max_staleness=100,
            push_interval_steps=10,
            last_mixed_payload_ids={},
        )

        self.assertEqual(result["mixed_peer_updates"], 1)
        self.assertEqual(result["max_staleness"], 10)
        self.assertTrue(
            torch.allclose(result["state"]["weight"], torch.tensor([4.5454545]))
        )

    def test_slow_peer_is_mixed_when_gap_matches_relative_speed(self) -> None:
        result = _mix_with_latest_peer_payloads(
            current_state={"weight": torch.tensor([0.0])},
            current_version=22000,
            local_sample_count=1,
            server=_FakeServer([_payload(version=6400)]),
            neighbor_ids=["peer"],
            base_alpha=1.0,
            max_staleness=5000,
            push_interval_steps=100,
            local_relative_speed=1.2,
            peer_relative_speeds={"peer": 0.35},
            last_mixed_payload_ids={},
        )

        self.assertEqual(result["mixed_peer_updates"], 1)
        self.assertEqual(result["dropped_peer_updates"], 0)
        self.assertEqual(result["stale_mixed_peer_updates"], 1)
        self.assertGreater(result["max_staleness"], 5000)
        self.assertLess(result["max_normalized_staleness"], 50)

    def test_nonfinite_merged_state_is_dropped(self) -> None:
        original_check_state_finite = async_gossip.check_state_finite

        def fake_check_state_finite(state):
            if torch.equal(state["weight"], torch.tensor([5.0])):
                return original_check_state_finite({"weight": torch.tensor([float("nan")])})
            return original_check_state_finite(state)

        async_gossip.check_state_finite = fake_check_state_finite
        try:
            result = _mix_with_latest_peer_payloads(
                current_state={"weight": torch.tensor([0.0])},
                current_version=10,
                local_sample_count=1,
                server=_FakeServer([_payload(version=10)]),
                neighbor_ids=["peer"],
                base_alpha=1.0,
                max_staleness=100,
                push_interval_steps=10,
                last_mixed_payload_ids={},
            )
        finally:
            async_gossip.check_state_finite = original_check_state_finite

        self.assertEqual(result["mixed_peer_updates"], 0)
        self.assertEqual(result["dropped_peer_updates"], 1)
        self.assertEqual(result["nonfinite_peer_updates"], 1)
        self.assertIn("nonfinite_merged_state", result["drop_reasons"][0])
        self.assertTrue(torch.equal(result["state"]["weight"], torch.tensor([0.0])))

    def test_exchange_rejects_nonfinite_local_state_before_sending(self) -> None:
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(float("nan"))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with self.assertRaisesRegex(ValueError, "outgoing async state"):
            _exchange_async_update(
                model=model,
                device=torch.device("cpu"),
                optimizer=optimizer,
                current_step=1,
                sample_count=1,
                run_id="run",
                self_node_id="local",
                server=_FakeServer([]),
                neighbors=[],
                push_fanout=0,
                push_interval_steps=1,
                base_alpha=1.0,
                max_staleness=10,
                transport_timeout_s=0.1,
                last_mixed_payload_ids={},
            )

    def test_successful_exchange_merge_clears_optimizer_state(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = _build_optimizer_with_state(model)
        self.assertGreater(len(optimizer.state), 0)
        peer_state = {
            "weight": torch.tensor([[10.0]]),
            "bias": torch.tensor([0.0]),
        }
        neighbor = SimpleNamespace(id="peer", host="127.0.0.1", port=1)
        original_push = async_gossip._push_payload_best_effort
        async_gossip._push_payload_best_effort = lambda **kwargs: PushAttemptResult(
            ok=True,
            target=kwargs["target"],
            elapsed_s=0.1,
            num_bytes=1,
        )
        try:
            result = _exchange_async_update(
                model=model,
                device=torch.device("cpu"),
                optimizer=optimizer,
                current_step=10,
                sample_count=1,
                run_id="run",
                self_node_id="local",
                server=_FakeServer([_payload(version=10, state=peer_state)]),
                neighbors=[neighbor],
                push_fanout=0,
                push_interval_steps=10,
                base_alpha=1.0,
                max_staleness=100,
                transport_timeout_s=0.1,
                last_mixed_payload_ids={},
            )
        finally:
            async_gossip._push_payload_best_effort = original_push

        self.assertEqual(result["mixed_peer_updates"], 1)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(result["pushes_sent"], 1)
        self.assertEqual(result["failed_pushes"], 0)
        self.assertEqual(result["push_elapsed_s"], 0.1)
        self.assertTrue(torch.allclose(model.weight.detach(), torch.tensor([[5.0]])))

    def test_exchange_without_merge_preserves_optimizer_state(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = _build_optimizer_with_state(model)
        state_count = len(optimizer.state)

        result = _exchange_async_update(
            model=model,
            device=torch.device("cpu"),
            optimizer=optimizer,
            current_step=10,
            sample_count=1,
            run_id="run",
            self_node_id="local",
            server=_FakeServer([]),
            neighbors=[],
            push_fanout=0,
            push_interval_steps=10,
            base_alpha=1.0,
            max_staleness=100,
            transport_timeout_s=0.1,
            last_mixed_payload_ids={},
        )

        self.assertEqual(result["mixed_peer_updates"], 0)
        self.assertEqual(len(optimizer.state), state_count)

    def test_exchange_records_push_failure_reason(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        neighbor = SimpleNamespace(id="peer", host="127.0.0.1", port=1)
        original_push = async_gossip._push_payload_best_effort
        async_gossip._push_payload_best_effort = lambda **kwargs: PushAttemptResult(
            ok=False,
            target=kwargs["target"],
            elapsed_s=0.2,
            error_type="preflight_ping_failed",
            error_message="UNAVAILABLE",
        )
        try:
            result = _exchange_async_update(
                model=model,
                device=torch.device("cpu"),
                optimizer=optimizer,
                current_step=10,
                sample_count=1,
                run_id="run",
                self_node_id="local",
                server=_FakeServer([]),
                neighbors=[neighbor],
                push_fanout=0,
                push_interval_steps=10,
                base_alpha=1.0,
                max_staleness=100,
                transport_timeout_s=900.0,
                last_mixed_payload_ids={},
            )
        finally:
            async_gossip._push_payload_best_effort = original_push

        self.assertEqual(result["pushes_sent"], 0)
        self.assertEqual(result["failed_pushes"], 1)
        self.assertEqual(result["push_failed_targets"], ["peer"])
        self.assertEqual(result["push_elapsed_s"], 0.2)
        self.assertIn("peer@127.0.0.1:1:preflight_ping_failed", result["push_failure_reasons"][0])

    def test_push_uses_full_timeout_after_successful_preflight(self) -> None:
        calls: dict[str, float] = {}

        class FakeClient:
            def __init__(self, target: str):
                self.target = target

            def ping(self, sender_node_id: str, timeout_s: float = 5.0) -> dict:
                calls["ping_timeout_s"] = timeout_s
                return {"receiver_node_id": "peer", "message": "pong"}

            def push_payload(self, payload: PeerPayload, timeout_s: float = 15.0):
                calls["push_timeout_s"] = timeout_s
                return SimpleNamespace(num_bytes=123)

            def close(self) -> None:
                calls["close_count"] = calls.get("close_count", 0) + 1

        original_client = async_gossip.PeerClient
        async_gossip.PeerClient = FakeClient
        try:
            result = _push_payload_best_effort(
                target="peer:55051",
                sender_node_id="local",
                payload=_payload(),
                timeout_s=900.0,
            )
        finally:
            async_gossip.PeerClient = original_client

        self.assertTrue(result.ok)
        self.assertEqual(result.num_bytes, 123)
        self.assertLessEqual(calls["ping_timeout_s"], 5.0)
        self.assertGreater(calls["push_timeout_s"], 800.0)


class AsyncGossipTests(unittest.TestCase):
    def test_async_smoke_completes_and_observes_staleness(self) -> None:
        result = run_async_smoke(peer_count=3, rounds=2)
        node_results = result["node_results"]

        self.assertEqual(len(node_results), 3)
        self.assertTrue(all(code == 0 for code in result["exit_codes"]))
        self.assertTrue(all("error" not in payload for payload in node_results.values()))
        self.assertTrue(
            all(payload["mixed_peer_updates_total"] > 0 for payload in node_results.values())
        )
        self.assertTrue(
            any(payload["max_observed_staleness"] > 0 for payload in node_results.values())
        )
        self.assertTrue(
            all(payload["received_payload_count"] >= 1 for payload in node_results.values())
        )
        self.assertTrue(
            all(payload["push_count_total"] > payload["epoch_count"] for payload in node_results.values())
        )
        self.assertTrue(
            all("macro_f1" in payload["final_test_metrics"] for payload in node_results.values())
        )
        self.assertTrue(
            all(payload["history_kind"] == "epochs" for payload in node_results.values())
        )
        self.assertTrue(
            all(payload["completion_reported"] for payload in node_results.values())
        )
        self.assertTrue(
            any(payload["control_plane_role"] == "bootstrap" for payload in node_results.values())
        )
        self.assertTrue(
            any(
                payload["control_plane_role"] == "bootstrap"
                and payload["completion_cluster_complete"] is True
                for payload in node_results.values()
            )
        )

    def test_async_smoke_runs_with_micro_shards(self) -> None:
        result = run_async_smoke(peer_count=3, rounds=2, storage_mode="micro_shards")
        node_results = result["node_results"]

        self.assertEqual(len(node_results), 3)
        self.assertTrue(all(code == 0 for code in result["exit_codes"]))
        self.assertTrue(
            all(payload["mixed_peer_updates_total"] > 0 for payload in node_results.values())
        )
        self.assertTrue(
            all(payload["push_count_total"] > payload["epoch_count"] for payload in node_results.values())
        )
        self.assertTrue(
            all("macro_f1" in payload["final_test_metrics"] for payload in node_results.values())
        )

    def test_async_smoke_runs_with_adaptive_micro_shards(self) -> None:
        result = run_async_smoke(
            peer_count=3,
            rounds=2,
            storage_mode="micro_shards",
            scheduler_mode="adaptive",
        )
        node_results = result["node_results"]

        self.assertEqual(len(node_results), 3)
        self.assertTrue(all(code == 0 for code in result["exit_codes"]))
        self.assertTrue(
            all(payload["mixed_peer_updates_total"] > 0 for payload in node_results.values())
        )
        self.assertTrue(
            all(payload["push_count_total"] > payload["epoch_count"] for payload in node_results.values())
        )
        self.assertTrue(
            all("macro_f1" in payload["final_test_metrics"] for payload in node_results.values())
        )
        self.assertTrue(
            all(
                payload["scheduler_history_file"] is not None
                and payload["scheduler_history_file"].endswith("scheduler_history.csv")
                for payload in node_results.values()
            )
        )

    def test_async_smoke_handles_empty_adaptive_windows(self) -> None:
        result = run_async_smoke(
            peer_count=3,
            rounds=4,
            storage_mode="micro_shards",
            scheduler_mode="adaptive",
        )
        node_results = result["node_results"]

        self.assertEqual(len(node_results), 3)
        self.assertTrue(all(code == 0 for code in result["exit_codes"]))
        self.assertTrue(
            all("error" not in payload for payload in node_results.values())
        )
        self.assertTrue(
            all(payload["epoch_count"] == 4 for payload in node_results.values())
        )
        self.assertTrue(
            all(payload["total_samples_processed"] > 0 for payload in node_results.values())
        )

    def test_async_smoke_reuses_dataset_every_epoch_in_adaptive_mode(self) -> None:
        result = run_async_smoke(
            peer_count=3,
            rounds=4,
            storage_mode="micro_shards",
            scheduler_mode="adaptive",
        )
        node_results = result["node_results"]

        self.assertEqual(len(node_results), 3)
        self.assertTrue(all(code == 0 for code in result["exit_codes"]))
        self.assertTrue(
            all("error" not in payload for payload in node_results.values())
        )
        self.assertTrue(
            all(len(payload["epochs"]) == 4 for payload in node_results.values())
        )
        self.assertTrue(
            all(
                all(epoch_row["samples_processed"] > 0 for epoch_row in payload["epochs"])
                for payload in node_results.values()
            )
        )

    def test_async_smoke_uses_multiple_adaptive_windows_per_epoch(self) -> None:
        result = run_async_smoke(
            peer_count=3,
            rounds=2,
            storage_mode="micro_shards",
            scheduler_mode="adaptive",
            rebalance_window_batches=1,
            fake_train_size=96,
        )
        node_results = result["node_results"]

        self.assertEqual(len(node_results), 3)
        self.assertTrue(all(code == 0 for code in result["exit_codes"]))
        self.assertTrue(
            all("error" not in payload for payload in node_results.values())
        )
        self.assertTrue(
            all(
                all(epoch_row["window_count"] > 1 for epoch_row in payload["epochs"])
                for payload in node_results.values()
            )
        )
        self.assertTrue(
            all(
                all(len(epoch_row.get("windows", [])) == epoch_row["window_count"] for epoch_row in payload["epochs"])
                for payload in node_results.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
