from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.config.loader import load_resolved_config, load_yaml
from decentr_my_own.data.shards import build_dataset_shards
from decentr_my_own.training.engine import LocalTrainOverrides, run_local_training


class LocalTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster = PROJECT_ROOT / "configs" / "cluster.example.yaml"
        self.training = PROJECT_ROOT / "configs" / "training.local-smoke.yaml"

    def test_local_training_creates_metrics_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_payload = load_yaml(self.training)
            training_payload["logging"]["log_dir"] = str(tmp_path / "logs")
            training_payload["logging"]["checkpoint_dir"] = str(tmp_path / "checkpoints")

            training_path = tmp_path / "training.yaml"
            with training_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(training_payload, handle, sort_keys=False)

            resolved = load_resolved_config(self.cluster, training_path, "node-1")
            result = run_local_training(
                resolved,
                LocalTrainOverrides(
                    epochs=1,
                    max_train_batches=2,
                    max_eval_batches=1,
                    run_name="unit-test-run",
                ),
            )

            self.assertEqual(len(result.epoch_history), 1)
            self.assertTrue((result.log_dir / "run_summary.json").exists())
            self.assertTrue((result.log_dir / "epoch_metrics.csv").exists())
            self.assertTrue((result.checkpoint_dir / "epoch-001.pt").exists())

            summary = json.loads((result.log_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["self_node_id"], "node-1")
            self.assertEqual(summary["dataset"], "FakeData")
            self.assertIn("macro_f1", summary["final_test_metrics"])

    def test_local_training_runs_with_micro_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_payload = load_yaml(self.training)
            training_payload["dataset"]["storage_mode"] = "micro_shards"
            training_payload["dataset"]["manifest_path"] = str(tmp_path / "manifest.json")
            training_payload["dataset"]["shard_samples"] = 8
            training_payload["logging"]["log_dir"] = str(tmp_path / "logs")
            training_payload["logging"]["checkpoint_dir"] = str(tmp_path / "checkpoints")

            training_path = tmp_path / "training.micro.yaml"
            with training_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(training_payload, handle, sort_keys=False)

            build_dataset_shards(load_resolved_config(self.cluster, training_path, "node-1").training)

            resolved = load_resolved_config(self.cluster, training_path, "node-1")
            result = run_local_training(
                resolved,
                LocalTrainOverrides(
                    epochs=1,
                    max_train_batches=2,
                    max_eval_batches=1,
                    run_name="micro-shard-unit-test",
                ),
            )

            self.assertEqual(len(result.epoch_history), 1)
            self.assertTrue((result.log_dir / "run_summary.json").exists())
            summary = json.loads((result.log_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["dataset"], "FakeData")
            self.assertIn("macro_f1", summary["final_test_metrics"])


if __name__ == "__main__":
    unittest.main()
