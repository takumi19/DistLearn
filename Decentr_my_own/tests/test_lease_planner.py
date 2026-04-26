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
        self.assertEqual(sum(len(shards) for shards in window0.assignments.values()), 12)

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

    def test_adaptive_planner_uses_heterogeneous_weights_before_reports(self) -> None:
        planner = AdaptiveLeasePlanner(
            manifest=_build_manifest(train_shards=12, shard_samples=4),
            cluster=self.cluster,
            training=_build_training_config(warmup_windows=0, min_local_shards=1),
        )

        window0 = planner.plan_window(window_id=0)

        self.assertGreater(len(window0.shards_for_node("node-2")), len(window0.shards_for_node("node-1")))
        self.assertGreater(len(window0.shards_for_node("node-1")), len(window0.shards_for_node("node-3")))

    def test_warmup_window_preserves_base_heterogeneous_weights(self) -> None:
        planner = AdaptiveLeasePlanner(
            manifest=_build_manifest(train_shards=12, shard_samples=4),
            cluster=self.cluster,
            training=_build_training_config(warmup_windows=2, min_local_shards=1),
        )

        window0 = planner.plan_window(window_id=0)
        self.assertGreater(len(window0.shards_for_node("node-2")), len(window0.shards_for_node("node-1")))
        self.assertGreater(len(window0.shards_for_node("node-1")), len(window0.shards_for_node("node-3")))

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
        self.assertGreater(len(window1.shards_for_node("node-2")), len(window1.shards_for_node("node-1")))
        self.assertGreater(len(window1.shards_for_node("node-1")), len(window1.shards_for_node("node-3")))

    def test_adaptive_planner_prefers_local_inventory_for_reuse(self) -> None:
        planner = AdaptiveLeasePlanner(
            manifest=_build_manifest(train_shards=6, shard_samples=4),
            cluster=self.cluster,
            training=_build_training_config(warmup_windows=0, min_local_shards=1),
        )

        first_plan = planner.plan_window(window_id=0)
        node_1_shards = tuple(first_plan.shards_for_node("node-1"))
        self.assertTrue(node_1_shards)

        second_plan = planner.plan_window(
            window_id=1,
            reports=[
                ThroughputReportRecord(
                    node_id="node-1",
                    window_id=0,
                    samples_processed=8,
                    window_seconds=1.0,
                    effective_throughput=8.0,
                    local_inventory=node_1_shards,
                ),
                ThroughputReportRecord(
                    node_id="node-2",
                    window_id=0,
                    samples_processed=8,
                    window_seconds=1.0,
                    effective_throughput=8.0,
                    local_inventory=tuple(first_plan.shards_for_node("node-2")),
                ),
                ThroughputReportRecord(
                    node_id="node-3",
                    window_id=0,
                    samples_processed=8,
                    window_seconds=1.0,
                    effective_throughput=8.0,
                    local_inventory=tuple(first_plan.shards_for_node("node-3")),
                ),
            ],
        )

        self.assertTrue(set(node_1_shards) & set(second_plan.shards_for_node("node-1")))

    def test_each_epoch_covers_full_dataset_and_resets_assignments(self) -> None:
        manifest = _build_manifest(train_shards=12, shard_samples=4)
        planner = AdaptiveLeasePlanner(
            manifest=manifest,
            cluster=self.cluster,
            training=_build_training_config(warmup_windows=0, min_local_shards=1),
        )

        manifest_shard_ids = {shard.shard_id for shard in manifest.shards}
        epoch_shard_sets = []
        for epoch_id in range(2):
            reports = None
            if epoch_id > 0:
                reports = [
                    ThroughputReportRecord(
                        node_id=node.id,
                        window_id=epoch_id - 1,
                        samples_processed=8,
                        window_seconds=1.0,
                        effective_throughput=8.0,
                    )
                    for node in self.cluster.nodes
                ]
            plan = planner.plan_window(window_id=epoch_id, reports=reports)
            epoch_shards = set()
            for node in self.cluster.nodes:
                epoch_shards.update(plan.shards_for_node(node.id))
            epoch_shard_sets.append(epoch_shards)

        self.assertEqual(epoch_shard_sets[0], manifest_shard_ids)
        self.assertEqual(epoch_shard_sets[1], manifest_shard_ids)

    def test_adaptive_planner_splits_epoch_into_multiple_windows(self) -> None:
        manifest = _build_manifest(train_shards=12, shard_samples=4)
        planner = AdaptiveLeasePlanner(
            manifest=manifest,
            cluster=self.cluster,
            training=_build_training_config(
                warmup_windows=0,
                min_local_shards=1,
                batch_size=2,
            ),
        )

        epoch_zero_plans = []
        window_id = 0
        while True:
            plan = planner.plan_window(window_id=window_id)
            if plan.epoch_id != 0:
                break
            epoch_zero_plans.append(plan)
            window_id += 1

        self.assertGreater(len(epoch_zero_plans), 1)
        self.assertTrue(all(plan.epoch_window_count == len(epoch_zero_plans) for plan in epoch_zero_plans))
        self.assertEqual(
            [plan.epoch_window_index for plan in epoch_zero_plans],
            list(range(len(epoch_zero_plans))),
        )

        seen_shard_ids = []
        for plan in epoch_zero_plans:
            for node in self.cluster.nodes:
                seen_shard_ids.extend(plan.shards_for_node(node.id))

        manifest_shard_ids = {shard.shard_id for shard in manifest.shards}
        self.assertEqual(set(seen_shard_ids), manifest_shard_ids)
        self.assertEqual(len(seen_shard_ids), len(manifest_shard_ids))

    def test_adaptive_planner_rebalances_within_same_epoch(self) -> None:
        planner = AdaptiveLeasePlanner(
            manifest=_build_manifest(train_shards=18, shard_samples=4),
            cluster=self.cluster,
            training=_build_training_config(
                warmup_windows=0,
                min_local_shards=1,
                batch_size=2,
                rebalance_window_batches=4,
                partitioning="homogeneous",
            ),
        )

        window0 = planner.plan_window(window_id=0)
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

        self.assertEqual(window0.epoch_id, 0)
        self.assertEqual(window1.epoch_id, 0)
        self.assertGreater(window1.epoch_window_count, 1)
        self.assertGreater(len(window1.shards_for_node("node-2")), len(window1.shards_for_node("node-1")))
        self.assertGreater(len(window1.shards_for_node("node-1")), len(window1.shards_for_node("node-3")))


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


def _build_training_config(
    *,
    warmup_windows: int,
    min_local_shards: int,
    batch_size: int = 64,
    rebalance_window_batches: int = 2,
    partitioning: str = "heterogeneous",
) -> TrainingConfig:
    return TrainingConfig(
        model=ModelConfig(num_classes=10),
        dataset=DatasetConfig(
            name="FakeData",
            storage_mode="micro_shards",
            manifest_path="./manifest.json",
            shard_samples=4,
            scheduler_mode="adaptive",
            rebalance_window_batches=rebalance_window_batches,
            throughput_ema=0.0,
            warmup_windows=warmup_windows,
            min_local_shards=min_local_shards,
            partitioning=partitioning,
        ),
        optimization={"batch_size": batch_size},
    )


if __name__ == "__main__":
    unittest.main()
