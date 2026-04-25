from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.config.loader import load_resolved_config, load_training_config, load_yaml
from decentr_my_own.data.loaders import build_local_dataloaders
from decentr_my_own.data.shards import build_dataset_shards


class MicroShardLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster_path = PROJECT_ROOT / "configs" / "cluster.example.yaml"
        self.training_path = PROJECT_ROOT / "configs" / "training.local-smoke.yaml"

    def _write_micro_shard_training(
        self,
        directory: Path,
        *,
        partitioning: str = "homogeneous",
    ) -> Path:
        payload = load_yaml(self.training_path)
        payload["dataset"]["storage_mode"] = "micro_shards"
        payload["dataset"]["manifest_path"] = str(directory / "manifest.json")
        payload["dataset"]["shard_samples"] = 4
        payload["dataset"]["partitioning"] = partitioning
        payload["dataset"]["fake_train_size"] = 24
        payload["dataset"]["fake_val_size"] = 8
        payload["dataset"]["fake_test_size"] = 8
        path = directory / "training.micro.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        return path

    def test_train_assignments_cover_all_samples_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_path = self._write_micro_shard_training(Path(tmp_dir), partitioning="homogeneous")
            training = load_training_config(training_path)
            build_dataset_shards(training)

            all_sample_ids: list[int] = []
            for node_id in ["node-1", "node-2", "node-3"]:
                resolved = load_resolved_config(self.cluster_path, training_path, node_id)
                dataloaders = build_local_dataloaders(resolved)
                sampler = dataloaders.train_sampler
                self.assertIsNotNone(sampler)
                sample_ids = sampler.assigned_sample_ids()
                self.assertEqual(len(sample_ids), len(set(sample_ids)))
                all_sample_ids.extend(sample_ids)

            self.assertEqual(len(all_sample_ids), 24)
            self.assertEqual(len(all_sample_ids), len(set(all_sample_ids)))
            self.assertEqual(set(all_sample_ids), set(range(24)))

    def test_heterogeneous_assignment_gives_more_samples_to_faster_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_path = self._write_micro_shard_training(
                Path(tmp_dir),
                partitioning="heterogeneous",
            )
            training = load_training_config(training_path)
            build_dataset_shards(training)

            counts: dict[str, int] = {}
            for node_id in ["node-1", "node-2", "node-3"]:
                resolved = load_resolved_config(self.cluster_path, training_path, node_id)
                dataloaders = build_local_dataloaders(resolved)
                counts[node_id] = len(dataloaders.train_sampler)

            self.assertGreater(counts["node-2"], counts["node-1"])
            self.assertGreater(counts["node-1"], counts["node-3"])

    def test_train_sampler_changes_iteration_order_between_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_path = self._write_micro_shard_training(Path(tmp_dir), partitioning="homogeneous")
            training = load_training_config(training_path)
            build_dataset_shards(training)

            resolved = load_resolved_config(self.cluster_path, training_path, "node-1")
            dataloaders = build_local_dataloaders(resolved)
            sampler = dataloaders.train_sampler

            sampler.set_epoch(0)
            first_epoch = sampler.assigned_indices()
            sampler.set_epoch(1)
            second_epoch = sampler.assigned_indices()

            self.assertEqual(len(first_epoch), len(second_epoch))
            self.assertNotEqual(first_epoch, second_epoch)

    def test_val_and_test_loaders_cover_full_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_path = self._write_micro_shard_training(Path(tmp_dir))
            training = load_training_config(training_path)
            build_dataset_shards(training)

            resolved = load_resolved_config(self.cluster_path, training_path, "node-1")
            dataloaders = build_local_dataloaders(resolved)

            self.assertEqual(len(dataloaders.val.dataset), 8)
            self.assertEqual(len(dataloaders.test.dataset), 8)


if __name__ == "__main__":
    unittest.main()
