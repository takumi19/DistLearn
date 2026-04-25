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
from decentr_my_own.config.loader import load_cluster_config, load_resolved_config, load_yaml
from decentr_my_own.data.partitioning import PartitionPlanner, PartitionedIndexSampler


class PartitioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster_path = PROJECT_ROOT / "configs" / "cluster.example.yaml"
        self.training_path = PROJECT_ROOT / "configs" / "training.example.yaml"
        self.cluster = load_cluster_config(self.cluster_path)

    def _make_resolved_with_strategy(self, strategy: str):
        training_payload = load_yaml(self.training_path)
        training_payload["dataset"]["partitioning"] = strategy
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "training.yaml"
            with tmp_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(training_payload, handle, sort_keys=False)
            yield load_resolved_config(self.cluster_path, tmp_path, "node-1")

    def test_homogeneous_plan_is_disjoint_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_training = Path(tmp_dir) / "training.yaml"
            payload = load_yaml(self.training_path)
            payload["dataset"]["partitioning"] = "homogeneous"
            with tmp_training.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

            resolved = load_resolved_config(self.cluster_path, tmp_training, "node-1")
            planner = PartitionPlanner(resolved.cluster, resolved.training)
            plan = planner.plan(total_size=97, shuffle_token=0)

        all_indices = []
        for node_id in plan.assignments:
            all_indices.extend(plan.node_indices(node_id))

        self.assertEqual(sorted(all_indices), list(range(97)))
        counts = plan.counts_by_node()
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_heterogeneous_plan_respects_weights(self) -> None:
        resolved = load_resolved_config(self.cluster_path, self.training_path, "node-1")
        planner = PartitionPlanner(resolved.cluster, resolved.training)
        plan = planner.plan(total_size=372, shuffle_token=0)

        counts = plan.counts_by_node()
        self.assertEqual(counts["node-1"], 100)
        self.assertEqual(counts["node-2"], 216)
        self.assertEqual(counts["node-3"], 56)

    def test_same_seed_same_plan_different_epoch_changes_assignment(self) -> None:
        resolved = load_resolved_config(self.cluster_path, self.training_path, "node-1")
        planner = PartitionPlanner(resolved.cluster, resolved.training)

        plan_a = planner.plan(total_size=128, shuffle_token=3)
        plan_b = planner.plan(total_size=128, shuffle_token=3)
        plan_c = planner.plan(total_size=128, shuffle_token=4)

        self.assertEqual(plan_a.counts_by_node(), plan_b.counts_by_node())
        self.assertEqual(plan_a.node_indices("node-1"), plan_b.node_indices("node-1"))
        self.assertNotEqual(plan_a.node_indices("node-1"), plan_c.node_indices("node-1"))

    def test_sampler_changes_with_epoch(self) -> None:
        resolved = load_resolved_config(self.cluster_path, self.training_path, "node-2")
        planner = PartitionPlanner(resolved.cluster, resolved.training)
        sampler = PartitionedIndexSampler(planner, "node-2", total_size=120)

        sampler.set_epoch(0)
        first_epoch = list(iter(sampler))
        sampler.set_epoch(1)
        second_epoch = list(iter(sampler))

        self.assertEqual(len(first_epoch), len(second_epoch))
        self.assertNotEqual(first_epoch, second_epoch)

    def test_cli_inspect_partition_outputs_json(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            exit_code = main(
                [
                    "inspect-partition",
                    "--cluster",
                    str(self.cluster_path),
                    "--training",
                    str(self.training_path),
                    "--self-node",
                    "node-1",
                    "--shuffle-token",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["node_id"], "node-1")
        self.assertEqual(payload["shuffle_token"], 2)
        self.assertIn("all_nodes", payload)


if __name__ == "__main__":
    unittest.main()
