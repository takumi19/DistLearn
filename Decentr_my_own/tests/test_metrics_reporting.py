from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.cli import main
from decentr_my_own.metrics.reporting import build_run_report, compare_run_reports


class MetricsReportingTests(unittest.TestCase):
    def test_build_run_report_aggregates_node_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir)
            _write_summary(
                log_root / "sync-run" / "node-1" / "sync_run_summary.json",
                _summary_payload(
                    run_id="sync-run",
                    node_id="node-1",
                    mode="sync",
                    algorithm="local_sgd",
                    duration=4.0,
                    throughput=16.0,
                    accuracy=0.70,
                    loss=1.2,
                    best_val=0.66,
                    total_samples=64,
                    digest="abc",
                ),
            )
            _write_summary(
                log_root / "sync-run" / "node-2" / "sync_run_summary.json",
                _summary_payload(
                    run_id="sync-run",
                    node_id="node-2",
                    mode="sync",
                    algorithm="local_sgd",
                    duration=5.0,
                    throughput=14.0,
                    accuracy=0.74,
                    loss=1.1,
                    best_val=0.69,
                    total_samples=64,
                    digest="abc",
                ),
            )

            report = build_run_report(log_root, "sync-run")

        self.assertEqual(report["node_count"], 2)
        self.assertEqual(report["total_samples_processed"], 128)
        self.assertEqual(report["effective_samples_per_s_total"], 30.0)
        self.assertAlmostEqual(report["final_test_accuracy_mean"], 0.72)
        self.assertTrue(report["consistent_final_state"])

    def test_compare_run_reports_outputs_deltas(self) -> None:
        baseline = {
            "run_id": "sync-run",
            "modes": ["sync"],
            "effective_samples_per_s_total": 30.0,
            "run_duration_s_max": 5.0,
            "final_test_accuracy_mean": 0.72,
            "final_test_loss_mean": 1.15,
            "best_val_accuracy_mean": 0.675,
            "total_samples_processed": 128,
        }
        candidate = {
            "run_id": "async-run",
            "modes": ["async"],
            "effective_samples_per_s_total": 33.0,
            "run_duration_s_max": 4.5,
            "final_test_accuracy_mean": 0.75,
            "final_test_loss_mean": 1.05,
            "best_val_accuracy_mean": 0.70,
            "total_samples_processed": 128,
        }

        comparison = compare_run_reports(baseline, candidate)

        self.assertEqual(comparison["winner"]["accuracy"], "async-run")
        self.assertEqual(comparison["winner"]["throughput"], "async-run")
        self.assertEqual(comparison["winner"]["duration"], "async-run")
        self.assertAlmostEqual(comparison["deltas"]["final_test_accuracy_mean"], 0.03)

    def test_cli_report_run_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_root = Path(tmp_dir)
            _write_summary(
                log_root / "local-run" / "node-1" / "run_summary.json",
                _summary_payload(
                    run_id="local-run",
                    node_id="node-1",
                    mode="sync",
                    algorithm="local_sgd",
                    duration=3.0,
                    throughput=20.0,
                    accuracy=0.81,
                    loss=0.9,
                    best_val=0.79,
                    total_samples=64,
                    digest=None,
                ),
            )
            buffer = StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "report-run",
                        "--log-root",
                        str(log_root),
                        "--run-id",
                        "local-run",
                    ]
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["run_id"], "local-run")
        self.assertEqual(payload["node_count"], 1)


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summary_payload(
    *,
    run_id: str,
    node_id: str,
    mode: str,
    algorithm: str,
    duration: float,
    throughput: float,
    accuracy: float,
    loss: float,
    best_val: float,
    total_samples: int,
    digest: str | None,
) -> dict:
    return {
        "run_id": run_id,
        "cluster_name": "unit-test-cluster",
        "self_node_id": node_id,
        "device": "cpu",
        "dataset": "FakeData",
        "model": "resnet18",
        "mode": mode,
        "algorithm": algorithm,
        "history_kind": "rounds",
        "started_at": "2026-03-10T00:00:00+00:00",
        "finished_at": "2026-03-10T00:00:05+00:00",
        "run_duration_s": duration,
        "round_count": 1,
        "total_samples_processed": total_samples,
        "effective_samples_per_s": throughput,
        "best_val_accuracy": best_val,
        "final_test_metrics": {
            "loss": loss,
            "accuracy": accuracy,
        },
        "final_state_digest": digest,
    }


if __name__ == "__main__":
    unittest.main()
