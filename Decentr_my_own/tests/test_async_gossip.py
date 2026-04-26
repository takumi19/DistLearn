from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.algorithms.async_gossip import run_async_smoke


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
