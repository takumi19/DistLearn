from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.config.loader import (
    load_cluster_config,
    load_resolved_config,
    load_training_config,
    load_yaml,
)


class ConfigLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster = PROJECT_ROOT / "configs" / "cluster.example.yaml"
        self.training = PROJECT_ROOT / "configs" / "training.example.yaml"
        self.wan_cluster = PROJECT_ROOT / "configs" / "cluster.wan-2node.example.yaml"
        self.wan_training_sync = PROJECT_ROOT / "configs" / "training.wan-sync.example.yaml"
        self.wan_training_async = PROJECT_ROOT / "configs" / "training.wan-async.example.yaml"

    def test_example_configs_load(self) -> None:
        resolved = load_resolved_config(self.cluster, self.training, "node-2")
        self.assertEqual(resolved.self_node.id, "node-2")
        self.assertEqual(resolved.training.mode, "sync")
        self.assertEqual(len(resolved.cluster.nodes), 3)

    def test_invalid_cluster_rejects_asymmetric_graph(self) -> None:
        payload = load_yaml(self.cluster)
        payload["nodes"][0]["neighbors"] = ["node-2"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "cluster.yaml"
            with tmp_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            with self.assertRaises(ValidationError):
                load_cluster_config(tmp_path)

    def test_wan_example_configs_load(self) -> None:
        resolved_sync = load_resolved_config(self.wan_cluster, self.wan_training_sync, "node-mac")
        resolved_async = load_resolved_config(
            self.wan_cluster, self.wan_training_async, "node-win"
        )

        self.assertEqual(resolved_sync.self_node.bind_host, "0.0.0.0")
        self.assertEqual(resolved_sync.training.mode, "sync")
        self.assertEqual(resolved_async.training.mode, "async")

    def test_micro_shards_requires_manifest_path(self) -> None:
        payload = load_yaml(self.training)
        payload["dataset"]["storage_mode"] = "micro_shards"
        payload["dataset"]["shard_samples"] = 8

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "training.yaml"
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            with self.assertRaises(ValidationError):
                load_training_config(path)

    def test_micro_shards_requires_shard_samples(self) -> None:
        payload = load_yaml(self.training)
        payload["dataset"]["storage_mode"] = "micro_shards"
        payload["dataset"]["manifest_path"] = "./artifacts/shards/manifest.json"

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "training.yaml"
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            with self.assertRaises(ValidationError):
                load_training_config(path)


if __name__ == "__main__":
    unittest.main()
