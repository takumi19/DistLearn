from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.config.loader import load_training_config, load_yaml
from decentr_my_own.data.manifest import load_manifest, save_manifest
from decentr_my_own.data.shard_store import ShardStore
from decentr_my_own.data.shards import build_dataset_shards


class ShardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training_path = PROJECT_ROOT / "configs" / "training.local-smoke.yaml"

    def _write_micro_shard_training(self, directory: Path) -> Path:
        payload = load_yaml(self.training_path)
        payload["dataset"]["storage_mode"] = "micro_shards"
        payload["dataset"]["manifest_path"] = str(directory / "manifest.json")
        payload["dataset"]["shard_samples"] = 4
        payload["dataset"]["fake_train_size"] = 17
        payload["dataset"]["fake_val_size"] = 8
        payload["dataset"]["fake_test_size"] = 9
        path = directory / "training.micro.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        return path

    def test_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training = load_training_config(self._write_micro_shard_training(Path(tmp_dir)))
            result = build_dataset_shards(training)
            manifest = load_manifest(result.manifest_path)
            copy_path = Path(tmp_dir) / "manifest.copy.json"

            save_manifest(manifest, copy_path)
            reloaded = load_manifest(copy_path)

            self.assertEqual(manifest.model_dump(), reloaded.model_dump())

    def test_shard_store_resolves_and_loads_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training = load_training_config(self._write_micro_shard_training(Path(tmp_dir)))
            result = build_dataset_shards(training)
            store = ShardStore.from_manifest_path(result.manifest_path)

            first_shard = store.list_shards("train")[0]
            resolved = store.resolve_path(first_shard)
            payload = store.load_shard(first_shard)

            self.assertTrue(resolved.is_absolute())
            self.assertTrue(resolved.exists())
            self.assertEqual(payload.images.dtype, torch.uint8)
            self.assertEqual(tuple(payload.images.shape[1:]), (3, 32, 32))
            self.assertEqual(payload.labels.dtype, torch.int64)
            self.assertEqual(len(payload.sample_ids), payload.images.size(0))

    def test_checksum_verification_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training = load_training_config(self._write_micro_shard_training(Path(tmp_dir)))
            result = build_dataset_shards(training)
            store = ShardStore.from_manifest_path(result.manifest_path)

            first_shard = store.list_shards("train")[0]
            path = store.resolve_path(first_shard)
            path.write_bytes(path.read_bytes() + b"corrupt")

            with self.assertRaises(ValueError):
                store.verify_shard(first_shard)

    def test_deterministic_build_keeps_manifest_and_hashes_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_path = self._write_micro_shard_training(Path(tmp_dir))
            training = load_training_config(training_path)

            first_result = build_dataset_shards(training)
            first_manifest = load_manifest(first_result.manifest_path)

            second_result = build_dataset_shards(training, force=True)
            second_manifest = load_manifest(second_result.manifest_path)

            self.assertEqual(first_result.to_dict(), second_result.to_dict())
            self.assertEqual(first_manifest.model_dump(), second_manifest.model_dump())

    def test_split_coverage_is_complete_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training = load_training_config(self._write_micro_shard_training(Path(tmp_dir)))
            result = build_dataset_shards(training)
            store = ShardStore.from_manifest_path(result.manifest_path)

            for split_name, expected_size in {"train": 17, "val": 8, "test": 9}.items():
                seen: list[int] = []
                for shard in store.list_shards(split_name):
                    seen.extend(store.load_shard(shard).sample_ids)

                self.assertEqual(len(seen), expected_size)
                self.assertEqual(len(seen), len(set(seen)))
                self.assertEqual(set(seen), set(range(expected_size)))

    def test_last_shard_can_be_shorter_than_shard_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training = load_training_config(self._write_micro_shard_training(Path(tmp_dir)))
            result = build_dataset_shards(training)
            manifest = load_manifest(result.manifest_path)

            train_shards = manifest.shards_for_split("train")
            self.assertEqual(train_shards[-1].sample_count, 1)

    def test_build_result_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training = load_training_config(self._write_micro_shard_training(Path(tmp_dir)))
            result = build_dataset_shards(training)
            payload = result.to_dict()

            self.assertEqual(json.loads(json.dumps(payload))["shard_count"], result.shard_count)

    def test_trim_cache_evicts_unprotected_train_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            training = load_training_config(self._write_micro_shard_training(Path(tmp_dir)))
            result = build_dataset_shards(training)
            store = ShardStore.from_manifest_path(result.manifest_path)

            train_shards = store.list_shards("train")
            protected = train_shards[0]
            cache_limit = protected.byte_size + 1

            evicted = store.trim_cache(
                max_cache_bytes=cache_limit,
                protected_shard_ids=(protected.shard_id,),
                split="train",
            )

            self.assertTrue(evicted)
            self.assertTrue(store.has_local_shard(protected))
            self.assertTrue(all(not store.has_local_shard(shard_id) for shard_id in evicted))
            self.assertLessEqual(store.local_cache_size_bytes("train"), cache_limit)
            self.assertEqual(len(store.list_shards("train")), len(train_shards))


if __name__ == "__main__":
    unittest.main()
