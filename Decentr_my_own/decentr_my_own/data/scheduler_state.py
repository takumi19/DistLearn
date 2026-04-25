from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThroughputReportRecord:
    node_id: str
    window_id: int
    samples_processed: int
    window_seconds: float
    effective_throughput: float
    local_inventory: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "window_id": self.window_id,
            "samples_processed": self.samples_processed,
            "window_seconds": self.window_seconds,
            "effective_throughput": self.effective_throughput,
            "local_inventory": list(self.local_inventory),
        }


@dataclass(frozen=True)
class LeasePlanRecord:
    window_id: int = 0
    assignments: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def shards_for_node(self, node_id: str) -> tuple[str, ...]:
        return self.assignments.get(node_id, ())

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "assignments": {
                node_id: list(shard_ids) for node_id, shard_ids in self.assignments.items()
            },
        }
