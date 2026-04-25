from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from decentr_my_own.config.models import ClusterConfig, TrainingConfig
from decentr_my_own.data.manifest import ShardMeta, SplitName
from decentr_my_own.data.partitioning import (
    allocate_weighted_counts,
    compute_effective_weights,
    shuffle_indices,
)
from decentr_my_own.data.shard_store import ShardStore


Transform = Callable[[torch.Tensor], torch.Tensor] | None


@dataclass(frozen=True)
class NodeShardAssignment:
    node_id: str
    shard_ids: list[str]
    sample_count: int
    effective_weight: float


@dataclass(frozen=True)
class StaticShardAssignmentPlan:
    split: SplitName
    total_size: int
    strategy: str
    shuffle_scope: str
    shuffle_token: int
    seed: int
    assignments: dict[str, NodeShardAssignment]

    def shard_ids_for_node(self, node_id: str) -> list[str]:
        try:
            return self.assignments[node_id].shard_ids
        except KeyError as exc:
            raise KeyError(f"Unknown node id in shard assignment plan: {node_id}") from exc


class ShardDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        store: ShardStore,
        *,
        split: SplitName,
        transform: Transform = None,
    ):
        self.store = store
        self.split = split
        self.transform = transform
        self.shards = store.list_shards(split)
        self._shard_ids = [shard.shard_id for shard in self.shards]
        self._shard_offsets: list[int] = []
        self._shard_ends: list[int] = []
        self._shard_id_to_range: dict[str, tuple[int, int]] = {}
        self._payload_cache: dict[str, object] = {}

        cursor = 0
        for shard in self.shards:
            self._shard_offsets.append(cursor)
            cursor += shard.sample_count
            self._shard_ends.append(cursor)
            self._shard_id_to_range[shard.shard_id] = (self._shard_offsets[-1], cursor)
        self._length = cursor

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < 0 or index >= self._length:
            raise IndexError(f"Sample index out of range: {index}")
        shard_meta, offset = self._locate(index)
        payload = self._load_payload(shard_meta.shard_id)
        image = payload.images[offset]
        if self.transform is not None:
            image = self.transform(image)
        target = int(payload.labels[offset].item())
        return image, target

    def shard_ids(self) -> list[str]:
        return list(self._shard_ids)

    def indices_for_shards(self, shard_ids: Iterable[str]) -> list[int]:
        indices: list[int] = []
        for shard_id in shard_ids:
            start, end = self._shard_id_to_range[shard_id]
            indices.extend(range(start, end))
        return indices

    def sample_id_for_index(self, index: int) -> int:
        shard_meta, offset = self._locate(index)
        payload = self._load_payload(shard_meta.shard_id)
        return int(payload.sample_ids[offset])

    def sample_ids_for_shards(self, shard_ids: Iterable[str]) -> list[int]:
        sample_ids: list[int] = []
        for shard_id in shard_ids:
            payload = self._load_payload(shard_id)
            sample_ids.extend(int(item) for item in payload.sample_ids)
        return sample_ids

    def _locate(self, index: int) -> tuple[ShardMeta, int]:
        shard_position = bisect_right(self._shard_ends, index)
        shard_meta = self.shards[shard_position]
        shard_start = self._shard_offsets[shard_position]
        return shard_meta, index - shard_start

    def _load_payload(self, shard_id: str):
        if shard_id not in self._payload_cache:
            self._payload_cache[shard_id] = self.store.load_shard(shard_id)
        return self._payload_cache[shard_id]


class StaticShardAssignmentPlanner:
    def __init__(
        self,
        cluster: ClusterConfig,
        training: TrainingConfig,
        shards: list[ShardMeta],
        *,
        split: SplitName = "train",
    ):
        self.cluster = cluster
        self.training = training
        self.shards = list(shards)
        self.split = split

    def plan(self, shuffle_token: int = 0) -> StaticShardAssignmentPlan:
        total_size = sum(shard.sample_count for shard in self.shards)
        ordered_nodes = [node.id for node in self.cluster.nodes]
        weights = compute_effective_weights(
            self.cluster.nodes,
            self.training.dataset.partitioning,
        )
        targets = allocate_weighted_counts(total_size, weights)
        assigned_samples = {node_id: 0 for node_id in ordered_nodes}
        assignments = {node_id: [] for node_id in ordered_nodes}
        node_order = {node_id: index for index, node_id in enumerate(ordered_nodes)}

        for shard in self._ordered_shards(shuffle_token):
            node_id = max(
                ordered_nodes,
                key=lambda current_node_id: (
                    targets[current_node_id] - assigned_samples[current_node_id],
                    -assigned_samples[current_node_id],
                    -node_order[current_node_id],
                ),
            )
            assignments[node_id].append(shard.shard_id)
            assigned_samples[node_id] += shard.sample_count

        total_weight = sum(weights.values()) or 1.0
        return StaticShardAssignmentPlan(
            split=self.split,
            total_size=total_size,
            strategy=self.training.dataset.partitioning,
            shuffle_scope=self.training.dataset.shuffle_scope,
            shuffle_token=shuffle_token,
            seed=self.training.seed,
            assignments={
                node_id: NodeShardAssignment(
                    node_id=node_id,
                    shard_ids=assignments[node_id],
                    sample_count=assigned_samples[node_id],
                    effective_weight=weights[node_id] / total_weight,
                )
                for node_id in ordered_nodes
            },
        )

    def _ordered_shards(self, shuffle_token: int) -> list[ShardMeta]:
        shard_order = shuffle_indices(
            total_size=len(self.shards),
            seed=self.training.seed,
            shuffle_scope=self.training.dataset.shuffle_scope,
            shuffle_token=shuffle_token,
        )
        return [self.shards[index] for index in shard_order]


class StaticShardAssignmentSampler(Sampler[int]):
    def __init__(
        self,
        planner: StaticShardAssignmentPlanner,
        dataset: ShardDataset,
        node_id: str,
        *,
        shuffle_token: int = 0,
    ):
        self.planner = planner
        self.dataset = dataset
        self.node_id = node_id
        self.shuffle_token = shuffle_token

    def set_epoch(self, epoch: int) -> None:
        self.shuffle_token = epoch

    def assigned_shard_ids(self) -> list[str]:
        plan = self.planner.plan(self.shuffle_token)
        return plan.shard_ids_for_node(self.node_id)

    def assigned_sample_ids(self) -> list[int]:
        return self.dataset.sample_ids_for_shards(self.assigned_shard_ids())

    def assigned_indices(self) -> list[int]:
        indices = self.dataset.indices_for_shards(self.assigned_shard_ids())
        shuffle_order = shuffle_indices(
            total_size=len(indices),
            seed=self.planner.training.seed,
            shuffle_scope=self.planner.training.dataset.shuffle_scope,
            shuffle_token=self.shuffle_token,
        )
        return [indices[index] for index in shuffle_order]

    def __iter__(self) -> Iterable[int]:
        return iter(self.assigned_indices())

    def __len__(self) -> int:
        plan = self.planner.plan(self.shuffle_token)
        return plan.assignments[self.node_id].sample_count


class FixedShardAssignmentSampler(Sampler[int]):
    def __init__(
        self,
        dataset: ShardDataset,
        training: TrainingConfig,
        shard_ids: Iterable[str],
        *,
        shuffle_token: int = 0,
    ):
        self.dataset = dataset
        self.training = training
        self._shard_ids = list(shard_ids)
        self.shuffle_token = shuffle_token

    def set_epoch(self, epoch: int) -> None:
        self.shuffle_token = epoch

    def assigned_shard_ids(self) -> list[str]:
        return list(self._shard_ids)

    def assigned_sample_ids(self) -> list[int]:
        return self.dataset.sample_ids_for_shards(self._shard_ids)

    def assigned_indices(self) -> list[int]:
        indices = self.dataset.indices_for_shards(self._shard_ids)
        shuffle_order = shuffle_indices(
            total_size=len(indices),
            seed=self.training.seed,
            shuffle_scope=self.training.dataset.shuffle_scope,
            shuffle_token=self.shuffle_token,
        )
        return [indices[index] for index in shuffle_order]

    def __iter__(self) -> Iterable[int]:
        return iter(self.assigned_indices())

    def __len__(self) -> int:
        return len(self.dataset.indices_for_shards(self._shard_ids))


class ShardWindowLoader(DataLoader):
    """Thin semantic wrapper for shard-backed dataloaders."""

    pass
