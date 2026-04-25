from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.algorithms.sync_barrier import run_sync_smoke


class SyncBarrierTests(unittest.TestCase):
    def test_sync_smoke_converges_to_same_state_digest(self) -> None:
        result = run_sync_smoke(peer_count=3, rounds=1)
        node_results = result["node_results"]

        self.assertEqual(len(node_results), 3)
        digests = {
            node_id: payload["final_state_digest"] for node_id, payload in node_results.items()
        }
        self.assertEqual(len(set(digests.values())), 1)
        self.assertTrue(all(code == 0 for code in result["exit_codes"]))
        self.assertTrue(
            all("macro_f1" in payload["final_test_metrics"] for payload in node_results.values())
        )

    def test_sync_smoke_runs_with_micro_shards(self) -> None:
        result = run_sync_smoke(peer_count=3, rounds=1, storage_mode="micro_shards")
        node_results = result["node_results"]

        self.assertEqual(len(node_results), 3)
        self.assertTrue(all(code == 0 for code in result["exit_codes"]))
        self.assertTrue(all(payload["train_sample_count"] > 0 for payload in node_results.values()))
        self.assertTrue(
            all("macro_f1" in payload["final_test_metrics"] for payload in node_results.values())
        )

    def test_sync_smoke_runs_with_adaptive_micro_shards(self) -> None:
        result = run_sync_smoke(
            peer_count=3,
            rounds=2,
            storage_mode="micro_shards",
            scheduler_mode="adaptive",
        )
        node_results = result["node_results"]

        self.assertEqual(len(node_results), 3)
        self.assertTrue(all(code == 0 for code in result["exit_codes"]))
        self.assertTrue(all(payload["train_sample_count"] > 0 for payload in node_results.values()))
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


if __name__ == "__main__":
    unittest.main()
