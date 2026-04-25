from __future__ import annotations

from dataclasses import dataclass

from decentr_my_own.config.models import ClusterConfig, TrainingConfig
from decentr_my_own.data.manifest import DatasetManifest, ShardMeta, SplitName
from decentr_my_own.data.partitioning import allocate_weighted_counts, shuffle_indices
from decentr_my_own.data.scheduler_state import LeasePlanRecord, ThroughputReportRecord
from decentr_my_own.data.shard_dataset import StaticShardAssignmentPlanner


def build_static_lease_plan(
    manifest: DatasetManifest,
    cluster: ClusterConfig,
    training: TrainingConfig,
    *,
    split: SplitName = "train",
    shuffle_token: int = 0,
    window_id: int = 0,
) -> LeasePlanRecord:
    planner = StaticShardAssignmentPlanner(
        cluster,
        training,
        manifest.shards_for_split(split),
        split=split,
    )
    plan = planner.plan(shuffle_token=shuffle_token)
    return LeasePlanRecord(
        window_id=window_id,
        assignments={
            node_id: tuple(assignment.shard_ids)
            for node_id, assignment in plan.assignments.items()
        },
    )


@dataclass
class AdaptiveLeasePlanner:
    manifest: DatasetManifest
    cluster: ClusterConfig
    training: TrainingConfig
    split: SplitName = "train"
    shuffle_token: int = 0

    def __post_init__(self) -> None:
        self._ordered_nodes = [node.id for node in self.cluster.nodes]
        ordered_indices = shuffle_indices(
            total_size=len(self.manifest.shards_for_split(self.split)),
            seed=self.training.seed,
            shuffle_scope=self.training.dataset.shuffle_scope,
            shuffle_token=self.shuffle_token,
        )
        split_shards = self.manifest.shards_for_split(self.split)
        self._remaining_shards = [split_shards[index] for index in ordered_indices]
        self._capacity_ema = {node_id: 1.0 for node_id in self._ordered_nodes}
        self._next_window_id = 0

    def plan_window(
        self,
        *,
        window_id: int,
        reports: list[ThroughputReportRecord] | None = None,
    ) -> LeasePlanRecord:
        if window_id != self._next_window_id:
            raise ValueError(
                f"AdaptiveLeasePlanner expected window_id={self._next_window_id}, got {window_id}"
            )

        if window_id > 0 and reports is not None:
            self._update_capacities(reports)

        selected_shards = self._take_window_shards()
        if not selected_shards:
            plan = LeasePlanRecord(
                window_id=window_id,
                assignments={node_id: tuple() for node_id in self._ordered_nodes},
            )
            self._next_window_id += 1
            return plan

        weights = self._window_weights(window_id)
        assignments = self._assign_window_shards(selected_shards, weights)
        self._next_window_id += 1
        return LeasePlanRecord(
            window_id=window_id,
            assignments=assignments,
        )

    @property
    def remaining_shard_count(self) -> int:
        return len(self._remaining_shards)

    @property
    def capacities(self) -> dict[str, float]:
        return dict(self._capacity_ema)

    def _update_capacities(self, reports: list[ThroughputReportRecord]) -> None:
        ema = self.training.dataset.throughput_ema
        report_by_node = {report.node_id: report for report in reports}
        for node_id in self._ordered_nodes:
            report = report_by_node.get(node_id)
            if report is None or report.effective_throughput <= 0:
                continue
            observed = max(report.effective_throughput, 1e-8)
            self._capacity_ema[node_id] = (
                ema * self._capacity_ema[node_id] + (1.0 - ema) * observed
            )

    def _take_window_shards(self) -> list[ShardMeta]:
        if not self._remaining_shards:
            return []

        target_shards = max(
            self.training.dataset.rebalance_window_batches * len(self._ordered_nodes),
            1,
        )
        min_total_shards = max(
            self.training.dataset.min_local_shards * len(self._ordered_nodes),
            1,
        )
        selected: list[ShardMeta] = []

        while self._remaining_shards and (
            len(selected) < target_shards or len(selected) < min_total_shards
        ):
            shard = self._remaining_shards.pop(0)
            selected.append(shard)
        return selected

    def _window_weights(self, window_id: int) -> dict[str, float]:
        if window_id < self.training.dataset.warmup_windows:
            return {node_id: 1.0 for node_id in self._ordered_nodes}
        return {
            node_id: max(capacity, 1e-8) for node_id, capacity in self._capacity_ema.items()
        }

    def _assign_window_shards(
        self,
        shards: list[ShardMeta],
        weights: dict[str, float],
    ) -> dict[str, tuple[str, ...]]:
        ordered_shards = sorted(
            shards,
            key=lambda shard: (shard.sample_count, shard.shard_id),
            reverse=True,
        )
        assignments = {node_id: [] for node_id in self._ordered_nodes}
        assigned_samples = {node_id: 0 for node_id in self._ordered_nodes}
        assigned_shards = {node_id: 0 for node_id in self._ordered_nodes}
        node_order = {node_id: index for index, node_id in enumerate(self._ordered_nodes)}
        min_local_shards = self.training.dataset.min_local_shards

        if min_local_shards > 0:
            while ordered_shards:
                pending_nodes = [
                    node_id
                    for node_id in self._ordered_nodes
                    if assigned_shards[node_id] < min_local_shards
                ]
                if not pending_nodes:
                    break
                if len(ordered_shards) < len(pending_nodes):
                    break
                for node_id in pending_nodes:
                    if not ordered_shards:
                        break
                    shard = ordered_shards.pop(0)
                    assignments[node_id].append(shard.shard_id)
                    assigned_samples[node_id] += shard.sample_count
                    assigned_shards[node_id] += 1

        extra_shard_targets = allocate_weighted_counts(len(ordered_shards), weights)
        assigned_extra_shards = {node_id: 0 for node_id in self._ordered_nodes}
        for shard in ordered_shards:
            node_id = max(
                self._ordered_nodes,
                key=lambda current_node_id: (
                    extra_shard_targets[current_node_id]
                    - assigned_extra_shards[current_node_id],
                    weights[current_node_id],
                    -assigned_samples[current_node_id],
                    -assigned_shards[current_node_id],
                    -node_order[current_node_id],
                ),
            )
            assignments[node_id].append(shard.shard_id)
            assigned_samples[node_id] += shard.sample_count
            assigned_shards[node_id] += 1
            assigned_extra_shards[node_id] += 1

        return {node_id: tuple(assignments[node_id]) for node_id in self._ordered_nodes}
