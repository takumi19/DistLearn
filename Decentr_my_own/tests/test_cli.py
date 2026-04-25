from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.cli import main
from decentr_my_own.config.loader import load_yaml


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster = PROJECT_ROOT / "configs" / "cluster.example.yaml"
        self.training = PROJECT_ROOT / "configs" / "training.example.yaml"
        self.wan_cluster = PROJECT_ROOT / "configs" / "cluster.wan-2node.example.yaml"
        self.wan_training = PROJECT_ROOT / "configs" / "training.wan-sync.example.yaml"

    def test_help_exits_cleanly(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            with self.assertRaises(SystemExit) as ctx:
                main(["--help"])

        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("validate-config", buffer.getvalue())

    def test_show_config_outputs_json(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = main(
                [
                    "show-config",
                    "--cluster",
                    str(self.cluster),
                    "--training",
                    str(self.training),
                    "--self-node",
                    "node-1",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["self_node_id"], "node-1")
        self.assertEqual(payload["cluster"]["cluster_name"], "decentr-lab")

    def test_launch_plan_outputs_json(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = main(
                [
                    "launch-plan",
                    "--cluster",
                    str(self.wan_cluster),
                    "--training",
                    str(self.wan_training),
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["cluster_name"], "decentr-wan-2node")
        self.assertEqual(len(payload["per_node"]), 2)

    def test_build_shards_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            training_payload = load_yaml(PROJECT_ROOT / "configs" / "training.local-smoke.yaml")
            training_payload["dataset"]["storage_mode"] = "micro_shards"
            training_payload["dataset"]["manifest_path"] = str(tmp_path / "manifest.json")
            training_payload["dataset"]["shard_samples"] = 5
            training_payload["dataset"]["fake_train_size"] = 16
            training_payload["dataset"]["fake_val_size"] = 8
            training_payload["dataset"]["fake_test_size"] = 8

            training_path = tmp_path / "training.micro.yaml"
            with training_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(training_payload, handle, sort_keys=False)

            buffer = StringIO()
            with redirect_stdout(buffer):
                exit_code = main(
                    [
                        "build-shards",
                        "--training",
                        str(training_path),
                    ]
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["manifest_path"], str(tmp_path / "manifest.json"))
        self.assertIn("split_counts", payload)
        self.assertGreater(payload["shard_count"], 0)


if __name__ == "__main__":
    unittest.main()
