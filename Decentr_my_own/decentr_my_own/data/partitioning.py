from __future__ import annotations

import random
from dataclasses import dataclass
from math import floor
from typing import Iterable

from torch.utils.data import Sampler

from decentr_my_own.config.models import ClusterConfig, NodeConfig, TrainingConfig


@dataclass(frozen=True)
class NodePartition:
    node_id: str
    count: int
    fraction: float
    effective_weight: float
    indices: list[int]


@dataclass(frozen=True)
class PartitionPlan:
    total_size: int
    strategy: str
    shuffle_scope: str
    shuffle_token: int
    seed: int
    assignments: dict[str, NodePartition]

    def node_indices(self, node_id: str) -> list[int]:
        try:
            return self.assignments[node_id].indices
        except KeyError as exc:
            raise KeyError(f"Unknown node id in partition plan: {node_id}") from exc

    def counts_by_node(self) -> dict[str, int]:
        return {node_id: partition.count for node_id, partition in self.assignments.items()}


class PartitionPlanner:
    def __init__(self, cluster: ClusterConfig, training: TrainingConfig):
        self.cluster = cluster
        self.training = training

    def plan(self, total_size: int, shuffle_token: int = 0) -> PartitionPlan:
        if total_size < 0:
            raise ValueError("total_size must be non-negative")

        ordered_nodes = list(self.cluster.nodes)
        effective_weights = compute_effective_weights(
            ordered_nodes, self.training.dataset.partitioning
        )
        counts = allocate_weighted_counts(total_size, effective_weights)
        shuffled = shuffle_indices(
            total_size=total_size,
            seed=self.training.seed,
            shuffle_scope=self.training.dataset.shuffle_scope,
            shuffle_token=shuffle_token,
        )

        assignments: dict[str, NodePartition] = {}
        cursor = 0
        total_weight = sum(effective_weights.values()) or 1.0
        for node in ordered_nodes:
            count = counts[node.id]
            indices = shuffled[cursor : cursor + count]
            cursor += count
            assignments[node.id] = NodePartition(
                node_id=node.id,
                count=count,
                fraction=(count / total_size) if total_size else 0.0,
                effective_weight=effective_weights[node.id] / total_weight,
                indices=indices,
            )

        return PartitionPlan(
            total_size=total_size,
            strategy=self.training.dataset.partitioning,
            shuffle_scope=self.training.dataset.shuffle_scope,
            shuffle_token=shuffle_token,
            seed=self.training.seed,
            assignments=assignments,
        )

    def describe_node(self, node_id: str, total_size: int, shuffle_token: int = 0) -> dict:
        plan = self.plan(total_size=total_size, shuffle_token=shuffle_token)
        node_partition = plan.assignments[node_id]
        return {
            "node_id": node_id,
            "strategy": plan.strategy,
            "shuffle_scope": plan.shuffle_scope,
            "shuffle_token": shuffle_token,
            "total_size": total_size,
            "node_count": len(plan.assignments),
            "self": _partition_to_dict(node_partition),
            "all_nodes": {
                node_key: _partition_to_dict(partition)
                for node_key, partition in plan.assignments.items()
            },
        }


class PartitionedIndexSampler(Sampler[int]):
    def __init__(
        self,
        planner: PartitionPlanner,
        node_id: str,
        total_size: int,
        *,
        shuffle_token: int = 0,
    ):
        self.planner = planner
        self.node_id = node_id
        self.total_size = total_size
        self.shuffle_token = shuffle_token

    def set_epoch(self, epoch: int) -> None:
        self.shuffle_token = epoch

    def __iter__(self) -> Iterable[int]:
        plan = self.planner.plan(self.total_size, shuffle_token=self.shuffle_token)
        return iter(plan.node_indices(self.node_id))

    def __len__(self) -> int:
        plan = self.planner.plan(self.total_size, shuffle_token=self.shuffle_token)
        return plan.assignments[self.node_id].count


def _partition_to_dict(partition: NodePartition) -> dict:
    return {
        "count": partition.count,
        "fraction": partition.fraction,
        "effective_weight": partition.effective_weight,
        "preview_indices": partition.indices[:10],
    }


def compute_effective_weights(
    nodes: list[NodeConfig], strategy: str
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for node in nodes:
        if strategy == "homogeneous":
            weights[node.id] = 1.0
        else:
            weights[node.id] = node.weight * node.resources.relative_speed
    return weights


def allocate_weighted_counts(total_size: int, weights: dict[str, float]) -> dict[str, int]:
    if total_size == 0:
        return {node_id: 0 for node_id in weights}

    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Sum of weights must be positive")

    raw_targets = {
        node_id: total_size * (weight / weight_sum) for node_id, weight in weights.items()
    }
    counts = {node_id: floor(target) for node_id, target in raw_targets.items()}
    assigned = sum(counts.values())
    remaining = total_size - assigned

    if remaining > 0:
        ranked = sorted(
            weights.keys(),
            key=lambda node_id: (raw_targets[node_id] - counts[node_id], node_id),
            reverse=True,
        )
        for node_id in ranked[:remaining]:
            counts[node_id] += 1

    return counts


def shuffle_indices(
    *,
    total_size: int,
    seed: int,
    shuffle_scope: str,
    shuffle_token: int,
) -> list[int]:
    indices = list(range(total_size))
    scope_offset = 0 if shuffle_scope == "global_seeded" else 10_000_019
    effective_seed = seed + scope_offset + (shuffle_token * 1_000_003)
    rng = random.Random(effective_seed)
    rng.shuffle(indices)
    return indices


def _effective_weights(
    nodes: list[NodeConfig], strategy: str
) -> dict[str, float]:
    return compute_effective_weights(nodes, strategy)


def _allocate_counts(total_size: int, weights: dict[str, float]) -> dict[str, int]:
    return allocate_weighted_counts(total_size, weights)


def _shuffle_indices(
    *,
    total_size: int,
    seed: int,
    shuffle_scope: str,
    shuffle_token: int,
) -> list[int]:
    return shuffle_indices(
        total_size=total_size,
        seed=seed,
        shuffle_scope=shuffle_scope,
        shuffle_token=shuffle_token,
    )
