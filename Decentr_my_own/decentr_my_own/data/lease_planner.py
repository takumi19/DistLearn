from __future__ import annotations

from dataclasses import dataclass

from decentr_my_own.config.models import ClusterConfig, TrainingConfig
from decentr_my_own.data.manifest import DatasetManifest, ShardMeta, SplitName
from decentr_my_own.data.partitioning import (
    allocate_weighted_counts,
    compute_effective_weights,
    shuffle_indices,
)
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
        epoch_id=window_id,
        epoch_window_index=0,
        epoch_window_count=1,
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
        self._split_shards = list(self.manifest.shards_for_split(self.split))
        self._base_weights = compute_effective_weights(
            self.cluster.nodes,
            self.training.dataset.partitioning,
        )
        self._capacity_ema = {
            node_id: float(self._base_weights[node_id]) for node_id in self._ordered_nodes
        }
        self._inventory_by_node = {node_id: set() for node_id in self._ordered_nodes}
        self._next_window_id = 0
        self._next_epoch_id = 0
        self._active_epoch_id: int | None = None
        self._active_epoch_windows: list[list[ShardMeta]] = []
        self._active_epoch_window_index = 0

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
            self._update_feedback(reports)

        epoch_window = self._next_epoch_window()
        if not epoch_window:
            plan = LeasePlanRecord(
                window_id=window_id,
                epoch_id=self._active_epoch_id or 0,
                epoch_window_index=self._active_epoch_window_index,
                epoch_window_count=max(len(self._active_epoch_windows), 1),
                assignments={node_id: tuple() for node_id in self._ordered_nodes},
            )
            self._next_window_id += 1
            return plan

        weights = self._window_weights(window_id)
        current_epoch_id = self._active_epoch_id or 0
        current_epoch_window_index = self._active_epoch_window_index
        current_epoch_window_count = max(len(self._active_epoch_windows), 1)
        assignments = self._assign_window_shards(epoch_window, weights)
        self._active_epoch_window_index += 1
        if self._active_epoch_window_index >= len(self._active_epoch_windows):
            self._active_epoch_id = None
            self._active_epoch_windows = []
            self._active_epoch_window_index = 0
        self._next_window_id += 1
        return LeasePlanRecord(
            window_id=window_id,
            epoch_id=current_epoch_id,
            epoch_window_index=current_epoch_window_index,
            epoch_window_count=current_epoch_window_count,
            assignments=assignments,
        )

    @property
    def remaining_shard_count(self) -> int:
        return len(self._split_shards)

    @property
    def capacities(self) -> dict[str, float]:
        return dict(self._capacity_ema)

    def _update_feedback(self, reports: list[ThroughputReportRecord]) -> None:
        ema = self.training.dataset.throughput_ema
        report_by_node = {report.node_id: report for report in reports}
        for node_id in self._ordered_nodes:
            report = report_by_node.get(node_id)
            if report is None:
                continue
            self._inventory_by_node[node_id] = set(report.local_inventory)
            if report.effective_throughput > 0:
                observed = max(report.effective_throughput, 1e-8)
                self._capacity_ema[node_id] = (
                    ema * self._capacity_ema[node_id] + (1.0 - ema) * observed
                )

    def _epoch_shards(self, epoch_id: int) -> list[ShardMeta]:
        if not self._split_shards:
            return []
        shard_order = shuffle_indices(
            total_size=len(self._split_shards),
            seed=self.training.seed,
            shuffle_scope=self.training.dataset.shuffle_scope,
            shuffle_token=self.shuffle_token + epoch_id,
        )
        return [self._split_shards[index] for index in shard_order]

    def _next_epoch_window(self) -> list[ShardMeta]:
        if self._active_epoch_id is None:
            self._start_next_epoch()
        if not self._active_epoch_windows:
            return []
        return list(self._active_epoch_windows[self._active_epoch_window_index])

    def _start_next_epoch(self) -> None:
        epoch_id = self._next_epoch_id
        epoch_shards = self._epoch_shards(epoch_id)
        self._active_epoch_id = epoch_id
        self._active_epoch_windows = self._chunk_epoch_shards(epoch_shards)
        self._active_epoch_window_index = 0
        self._next_epoch_id += 1

    def _chunk_epoch_shards(self, epoch_shards: list[ShardMeta]) -> list[list[ShardMeta]]:
        if not epoch_shards:
            return []

        sample_budget = self._window_sample_budget()
        if sample_budget <= 0:
            return [list(epoch_shards)]

        windows: list[list[ShardMeta]] = []
        current_window: list[ShardMeta] = []
        current_window_samples = 0
        for shard in epoch_shards:
            if current_window and current_window_samples + shard.sample_count > sample_budget:
                windows.append(current_window)
                current_window = []
                current_window_samples = 0
            current_window.append(shard)
            current_window_samples += shard.sample_count

        if current_window:
            windows.append(current_window)
        return windows

    def _window_sample_budget(self) -> int:
        if not self._split_shards:
            return 0
        local_batch_budget = (
            self.training.dataset.rebalance_window_batches
            * self.training.optimization.batch_size
        )
        cluster_sample_budget = local_batch_budget * max(len(self._ordered_nodes), 1)
        min_shard_size = max((shard.sample_count for shard in self._split_shards), default=1)
        return max(cluster_sample_budget, min_shard_size)

    def _window_weights(self, window_id: int) -> dict[str, float]:
        if window_id < self.training.dataset.warmup_windows:
            return {
                node_id: max(weight, 1e-8)
                for node_id, weight in self._base_weights.items()
            }
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
        inventory_by_node = {
            node_id: set(shard_ids) for node_id, shard_ids in self._inventory_by_node.items()
        }

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
                    1 if shard.shard_id in inventory_by_node.get(current_node_id, set()) else 0,
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
