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
            all(payload["push_count_total"] > payload["round_count"] for payload in node_results.values())
        )
        self.assertTrue(
            all("macro_f1" in payload["final_test_metrics"] for payload in node_results.values())
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
            all(payload["push_count_total"] > payload["round_count"] for payload in node_results.values())
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
            all(payload["push_count_total"] > payload["round_count"] for payload in node_results.values())
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
            all(payload["round_count"] == 4 for payload in node_results.values())
        )


if __name__ == "__main__":
    unittest.main()
