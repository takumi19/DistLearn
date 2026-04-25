from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.config.loader import load_cluster_config
from decentr_my_own.config.models import DatasetConfig, ModelConfig, TrainingConfig
from decentr_my_own.data.lease_planner import AdaptiveLeasePlanner
from decentr_my_own.data.manifest import DatasetManifest, ShardMeta, SplitSummary
from decentr_my_own.data.scheduler_state import ThroughputReportRecord


class LeasePlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster = load_cluster_config(PROJECT_ROOT / "configs" / "cluster.example.yaml")

    def test_adaptive_planner_gives_more_future_work_to_fast_node(self) -> None:
        planner = AdaptiveLeasePlanner(
            manifest=_build_manifest(train_shards=12, shard_samples=4),
            cluster=self.cluster,
            training=_build_training_config(warmup_windows=0, min_local_shards=1),
        )

        window0 = planner.plan_window(window_id=0)
        self.assertEqual(sum(len(shards) for shards in window0.assignments.values()), 6)

        window1 = planner.plan_window(
            window_id=1,
            reports=[
                ThroughputReportRecord(
                    node_id="node-1",
                    window_id=0,
                    samples_processed=8,
                    window_seconds=1.0,
                    effective_throughput=8.0,
                ),
                ThroughputReportRecord(
                    node_id="node-2",
                    window_id=0,
                    samples_processed=24,
                    window_seconds=1.0,
                    effective_throughput=24.0,
                ),
                ThroughputReportRecord(
                    node_id="node-3",
                    window_id=0,
                    samples_processed=4,
                    window_seconds=1.0,
                    effective_throughput=4.0,
                ),
            ],
        )

        self.assertGreater(len(window1.shards_for_node("node-2")), len(window1.shards_for_node("node-1")))
        self.assertGreater(len(window1.shards_for_node("node-1")), len(window1.shards_for_node("node-3")))

    def test_warmup_window_stays_uniform_even_with_skewed_reports(self) -> None:
        planner = AdaptiveLeasePlanner(
            manifest=_build_manifest(train_shards=9, shard_samples=4),
            cluster=self.cluster,
            training=_build_training_config(warmup_windows=2, min_local_shards=1),
        )

        window0 = planner.plan_window(window_id=0)
        counts0 = [len(window0.shards_for_node(node.id)) for node in self.cluster.nodes]
        self.assertLessEqual(max(counts0) - min(counts0), 1)

        window1 = planner.plan_window(
            window_id=1,
            reports=[
                ThroughputReportRecord(
                    node_id="node-1",
                    window_id=0,
                    samples_processed=4,
                    window_seconds=1.0,
                    effective_throughput=4.0,
                ),
                ThroughputReportRecord(
                    node_id="node-2",
                    window_id=0,
                    samples_processed=40,
                    window_seconds=1.0,
                    effective_throughput=40.0,
                ),
                ThroughputReportRecord(
                    node_id="node-3",
                    window_id=0,
                    samples_processed=4,
                    window_seconds=1.0,
                    effective_throughput=4.0,
                ),
            ],
        )
        counts1 = [len(window1.shards_for_node(node.id)) for node in self.cluster.nodes]
        self.assertLessEqual(max(counts1) - min(counts1), 1)

    def test_plans_do_not_reuse_shards_across_windows(self) -> None:
        planner = AdaptiveLeasePlanner(
            manifest=_build_manifest(train_shards=12, shard_samples=4),
            cluster=self.cluster,
            training=_build_training_config(warmup_windows=0, min_local_shards=1),
        )

        all_shards: list[str] = []
        for window_id in range(2):
            reports = None
            if window_id > 0:
                reports = [
                    ThroughputReportRecord(
                        node_id=node.id,
                        window_id=window_id - 1,
                        samples_processed=8,
                        window_seconds=1.0,
                        effective_throughput=8.0,
                    )
                    for node in self.cluster.nodes
                ]
            plan = planner.plan_window(window_id=window_id, reports=reports)
            for node in self.cluster.nodes:
                all_shards.extend(plan.shards_for_node(node.id))

        self.assertEqual(len(all_shards), len(set(all_shards)))


def _build_manifest(*, train_shards: int, shard_samples: int) -> DatasetManifest:
    shards = []
    for index in range(train_shards):
        shards.append(
            ShardMeta(
                shard_id=f"train-{index:05d}",
                split="train",
                relative_path=f"manifest_shards/train/train-{index:05d}.pt",
                sample_count=shard_samples,
                byte_size=32,
                sha256="0" * 64,
            )
        )
    return DatasetManifest(
        dataset_name="FakeData",
        seed=7,
        shard_samples=shard_samples,
        num_classes=10,
        image_shape=(3, 32, 32),
        splits={
            "train": SplitSummary(split="train", sample_count=train_shards * shard_samples, shard_count=train_shards),
            "val": SplitSummary(split="val", sample_count=0, shard_count=0),
            "test": SplitSummary(split="test", sample_count=0, shard_count=0),
        },
        shards=shards,
    )


def _build_training_config(*, warmup_windows: int, min_local_shards: int) -> TrainingConfig:
    return TrainingConfig(
        model=ModelConfig(num_classes=10),
        dataset=DatasetConfig(
            name="FakeData",
            storage_mode="micro_shards",
            manifest_path="./manifest.json",
            shard_samples=4,
            scheduler_mode="adaptive",
            rebalance_window_batches=2,
            throughput_ema=0.0,
            warmup_windows=warmup_windows,
            min_local_shards=min_local_shards,
        ),
    )


if __name__ == "__main__":
    unittest.main()
