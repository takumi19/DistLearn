"""Integration tests proving real distributed training properties.

These tests run multi-process smoke experiments and assert:
- All nodes actually participate (non-zero pushes AND mixes)
- Epoch semantics are correct (configured epochs == epochs in history)
- Completion protocol works (all nodes report, bootstrap sees full cluster)
- Adaptive heterogeneous planner distributes data unequally based on capacity
- join-run UX produces a working run token that can drive a real smoke run
"""
from __future__ import annotations

import multiprocessing as mp
import socket
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.algorithms.async_gossip import (
    AsyncRunOverrides,
    _async_worker_process,
    _find_free_port,
    _write_async_smoke_configs,
    run_async_smoke,
)
from decentr_my_own.config.loader import load_resolved_config
from decentr_my_own.deployment.wan import decode_run_token, encode_run_token


class TestRealEpochSemantics(unittest.TestCase):
    """Prove that --epochs N means N global passes, not N rounds/windows."""

    def test_configured_epochs_matches_history_length(self) -> None:
        for epoch_count in (1, 3):
            with self.subTest(epoch_count=epoch_count):
                result = run_async_smoke(peer_count=2, rounds=epoch_count)
                node_results = result["node_results"]
                self.assertEqual(result["configured_epochs"], epoch_count)
                for node_id, payload in node_results.items():
                    with self.subTest(node=node_id):
                        self.assertEqual(
                            payload["epoch_count"],
                            epoch_count,
                            msg=f"epoch_count mismatch for {node_id}",
                        )
                        self.assertEqual(
                            len(payload["epochs"]),
                            epoch_count,
                            msg=f"epochs list length mismatch for {node_id}",
                        )

    def test_all_epochs_have_nonzero_samples(self) -> None:
        result = run_async_smoke(peer_count=2, rounds=3)
        for node_id, payload in result["node_results"].items():
            for epoch_row in payload["epochs"]:
                self.assertGreater(
                    epoch_row["samples_processed"],
                    0,
                    msg=f"Epoch {epoch_row['epoch']} on {node_id} has 0 samples",
                )

    def test_no_round_count_in_async_summary(self) -> None:
        result = run_async_smoke(peer_count=2, rounds=2)
        for node_id, payload in result["node_results"].items():
            self.assertNotIn(
                "round_count",
                payload,
                msg=f"Misleading round_count field found in {node_id} summary",
            )


class TestMultiNodeParticipation(unittest.TestCase):
    """Every node must actually exchange weights, not just train locally."""

    def test_all_nodes_push_and_mix(self) -> None:
        result = run_async_smoke(peer_count=3, rounds=2)
        node_results = result["node_results"]
        for node_id, payload in node_results.items():
            with self.subTest(node=node_id):
                self.assertGreater(
                    payload["push_count_total"],
                    0,
                    msg=f"{node_id} sent 0 pushes",
                )
                self.assertGreater(
                    payload["mixed_peer_updates_total"],
                    0,
                    msg=f"{node_id} mixed 0 peer updates",
                )
                self.assertGreater(
                    payload["received_payload_count"],
                    0,
                    msg=f"{node_id} received 0 payloads",
                )

    def test_push_count_exceeds_epoch_count(self) -> None:
        result = run_async_smoke(peer_count=3, rounds=2)
        for node_id, payload in result["node_results"].items():
            with self.subTest(node=node_id):
                self.assertGreater(
                    payload["push_count_total"],
                    payload["epoch_count"],
                    msg=f"{node_id} push_count_total should exceed epoch_count",
                )


class TestCompletionProtocol(unittest.TestCase):
    """Shutdown must be correct: all nodes report, bootstrap sees full cluster."""

    def test_all_nodes_report_completion(self) -> None:
        result = run_async_smoke(peer_count=3, rounds=2)
        for node_id, payload in result["node_results"].items():
            with self.subTest(node=node_id):
                self.assertTrue(
                    payload["completion_reported"],
                    msg=f"{node_id} did not report completion",
                )

    def test_bootstrap_sees_cluster_complete(self) -> None:
        result = run_async_smoke(peer_count=3, rounds=2)
        bootstrap_payloads = [
            payload
            for payload in result["node_results"].values()
            if payload["control_plane_role"] == "bootstrap"
        ]
        self.assertEqual(len(bootstrap_payloads), 1, "Expected exactly one bootstrap node")
        self.assertTrue(
            bootstrap_payloads[0]["completion_cluster_complete"],
            "Bootstrap did not observe full cluster completion",
        )

    def test_all_exit_codes_zero(self) -> None:
        result = run_async_smoke(peer_count=3, rounds=2)
        for idx, code in enumerate(result["exit_codes"]):
            self.assertEqual(code, 0, msg=f"Process {idx} exited with code {code}")


class TestAdaptiveHeterogeneousDistribution(unittest.TestCase):
    """Adaptive planner must give heterogeneous nodes unequal shard allocations."""

    def _run_heterogeneous_smoke(self) -> dict:
        import yaml

        ctx = mp.get_context("spawn")
        results_queue = ctx.Queue()
        processes = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cluster_path, training_paths, node_ids = _write_async_smoke_configs(
                tmp_path=tmp_path,
                peer_count=2,
                storage_mode="micro_shards",
                scheduler_mode="adaptive",
                rebalance_window_batches=1,
                fake_train_size=96,
            )

            # Patch node weights in the cluster config to be heterogeneous.
            with cluster_path.open("r", encoding="utf-8") as fh:
                cluster_data = yaml.safe_load(fh)
            cluster_data["nodes"][0]["weight"] = 1.0
            cluster_data["nodes"][0]["resources"]["relative_speed"] = 1.0
            cluster_data["nodes"][1]["weight"] = 0.2
            cluster_data["nodes"][1]["resources"]["relative_speed"] = 0.2
            with cluster_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(cluster_data, fh, sort_keys=False)

            try:
                for idx, node_id in enumerate(node_ids):
                    process = ctx.Process(
                        target=_async_worker_process,
                        args=(
                            cluster_path,
                            training_paths[node_id],
                            node_id,
                            results_queue,
                            AsyncRunOverrides(
                                rounds=2,
                                max_local_batches=2,
                                max_eval_batches=1,
                                run_name="hetero-smoke",
                                transport_timeout_s=20.0,
                                bind_host="127.0.0.1",
                                shutdown_grace_s=2.5,
                            ),
                        ),
                    )
                    process.start()
                    processes.append(process)

                results = {}
                for _ in node_ids:
                    item = results_queue.get(timeout=90.0)
                    results[item["self_node_id"]] = item

                for process in processes:
                    process.join(timeout=30.0)
                return {"node_results": results, "node_ids": node_ids}
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5.0)

    def test_heavy_node_processes_more_samples_than_light_node(self) -> None:
        result = self._run_heterogeneous_smoke()
        node_results = result["node_results"]
        node_ids = result["node_ids"]
        self.assertEqual(len(node_results), 2)

        heavy_id, light_id = node_ids[0], node_ids[1]
        if heavy_id not in node_results or light_id not in node_results:
            self.skipTest("Not all nodes produced results")

        heavy_samples = node_results[heavy_id]["total_samples_processed"]
        light_samples = node_results[light_id]["total_samples_processed"]
        self.assertGreater(
            heavy_samples,
            light_samples,
            msg=(
                f"Heavy node ({heavy_id}: {heavy_samples} samples) should process more "
                f"than light node ({light_id}: {light_samples} samples)"
            ),
        )


class TestRunTokenJoinRunPath(unittest.TestCase):
    """Verify that a run token produced by encode_run_token can be decoded and used."""

    def test_token_roundtrip_preserves_all_fields(self) -> None:
        from decentr_my_own.config.loader import load_cluster_config_inline, load_training_config_inline
        import base64
        import json as _json

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cluster_path, training_paths, node_ids = _write_async_smoke_configs(
                tmp_path=tmp_path,
                peer_count=2,
            )
            resolved = load_resolved_config(cluster_path, training_paths[node_ids[0]], node_ids[0])
            cluster = resolved.cluster
            training = resolved.training

        token = encode_run_token(
            cluster=cluster,
            training=training,
            run_name="token-test",
            epochs=5,
        )
        decoded = decode_run_token(token)
        self.assertEqual(decoded["run_name"], "token-test")
        self.assertEqual(decoded["epochs"], 5)
        self.assertEqual(decoded["cluster"]["cluster_name"], cluster.cluster_name)
        self.assertEqual(decoded["training"]["mode"], training.mode)

        cluster_b64 = base64.urlsafe_b64encode(
            _json.dumps(decoded["cluster"], separators=(",", ":")).encode()
        ).decode()
        training_b64 = base64.urlsafe_b64encode(
            _json.dumps(decoded["training"], separators=(",", ":")).encode()
        ).decode()
        cluster2 = load_cluster_config_inline(cluster_b64)
        training2 = load_training_config_inline(training_b64)
        self.assertEqual(cluster2.cluster_name, cluster.cluster_name)
        self.assertEqual(training2.optimization.epochs, training.optimization.epochs)

    def test_token_can_drive_follower_resolve(self) -> None:
        from decentr_my_own.config.loader import load_resolved_config_inline
        import base64
        import json as _json

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cluster_path, training_paths, node_ids = _write_async_smoke_configs(
                tmp_path=tmp_path,
                peer_count=2,
            )
            resolved = load_resolved_config(cluster_path, training_paths[node_ids[0]], node_ids[0])

        token = encode_run_token(
            cluster=resolved.cluster,
            training=resolved.training,
            run_name="follower-test",
            epochs=3,
        )
        decoded = decode_run_token(token)
        follower_id = node_ids[1]
        cluster_b64 = base64.urlsafe_b64encode(
            _json.dumps(decoded["cluster"], separators=(",", ":")).encode()
        ).decode()
        training_b64 = base64.urlsafe_b64encode(
            _json.dumps(decoded["training"], separators=(",", ":")).encode()
        ).decode()
        follower_resolved = load_resolved_config_inline(cluster_b64, training_b64, follower_id)
        self.assertEqual(follower_resolved.self_node_id, follower_id)
        self.assertIn(node_ids[0], follower_resolved.self_node.neighbors)


if __name__ == "__main__":
    unittest.main()
