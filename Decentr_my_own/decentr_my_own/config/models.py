from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PlatformName = Literal["macos", "windows", "linux", "unknown"]
AcceleratorName = Literal["cpu", "cuda", "mps"]


class NodeResources(BaseModel):
    cpu_cores: int = Field(default=4, ge=1)
    accelerator: AcceleratorName = "cpu"
    relative_speed: float = Field(default=1.0, gt=0)


class NodeConfig(BaseModel):
    id: str = Field(min_length=1)
    host: str = Field(min_length=1)
    bind_host: str = Field(default="0.0.0.0", min_length=1)
    port: int = Field(ge=1, le=65535)
    platform: PlatformName = "unknown"
    neighbors: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0)
    resources: NodeResources = Field(default_factory=NodeResources)

    @field_validator("neighbors")
    @classmethod
    def validate_neighbor_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Neighbor ids must be unique per node")
        return value


class ClusterConfig(BaseModel):
    cluster_name: str = Field(min_length=1)
    transport: Literal["grpc"] = "grpc"
    overlay_network: Literal["tailscale", "none"] = "tailscale"
    tls_enabled: bool = False
    bootstrap_node_id: str | None = None
    nodes: list[NodeConfig] = Field(min_length=1)

    def get_node(self, node_id: str) -> NodeConfig:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"Unknown node id: {node_id}")

    @model_validator(mode="after")
    def validate_graph(self) -> "ClusterConfig":
        ids = [node.id for node in self.nodes]
        known_ids = set(ids)
        if len(ids) != len(known_ids):
            raise ValueError("Node ids must be unique")

        if self.bootstrap_node_id is not None and self.bootstrap_node_id not in known_ids:
            raise ValueError("bootstrap_node_id must reference an existing node")

        adjacency = {node.id: set(node.neighbors) for node in self.nodes}
        for node in self.nodes:
            if len(self.nodes) > 1 and not node.neighbors:
                raise ValueError(f"Node '{node.id}' must have at least one neighbor")
            if node.id in adjacency[node.id]:
                raise ValueError(f"Node '{node.id}' cannot reference itself as a neighbor")

            unknown = adjacency[node.id] - known_ids
            if unknown:
                unknown_ids = ", ".join(sorted(unknown))
                raise ValueError(
                    f"Node '{node.id}' references unknown neighbors: {unknown_ids}"
                )

        for node in self.nodes:
            for neighbor_id in adjacency[node.id]:
                if node.id not in adjacency[neighbor_id]:
                    raise ValueError(
                        "Neighbor relations must be symmetric for the fixed graph: "
                        f"'{node.id}' -> '{neighbor_id}' is missing the reverse edge"
                    )

        return self


class ModelConfig(BaseModel):
    name: str = Field(default="resnet18", min_length=1)
    num_classes: int = Field(default=100, ge=2)
    normalization: Literal["groupnorm", "batchnorm", "frozen_batchnorm"] = "groupnorm"


class DatasetConfig(BaseModel):
    name: str = Field(default="CIFAR100", min_length=1)
    root: str = "./data"
    storage_mode: Literal["replicated", "micro_shards"] = "replicated"
    manifest_path: str | None = None
    shard_samples: int | None = Field(default=None, ge=1)
    cache_dir: str | None = None
    max_cache_bytes: int = Field(default=1_073_741_824, ge=1)
    prefetch_shards: int = Field(default=1, ge=0)
    transfer_chunk_bytes: int = Field(default=1_048_576, ge=1)
    scheduler_mode: Literal["static", "adaptive"] = "static"
    rebalance_window_batches: int = Field(default=50, ge=1)
    throughput_ema: float = Field(default=0.9, ge=0.0, lt=1.0)
    warmup_windows: int = Field(default=1, ge=0)
    min_local_shards: int = Field(default=1, ge=0)
    transfer_policy: Literal["pull"] = "pull"
    distributed_eval: bool = False
    partitioning: Literal["homogeneous", "heterogeneous"] = "heterogeneous"
    shuffle_scope: Literal["global_seeded", "round_seeded"] = "global_seeded"
    val_split: float = Field(default=0.1, gt=0, lt=0.5)
    num_workers: int = Field(default=0, ge=0)
    download: bool = True
    image_size: int = Field(default=32, ge=8)
    fake_train_size: int = Field(default=256, ge=16)
    fake_val_size: int = Field(default=64, ge=8)
    fake_test_size: int = Field(default=64, ge=8)

    @model_validator(mode="after")
    def validate_micro_shards(self) -> "DatasetConfig":
        if self.storage_mode == "micro_shards":
            if not self.manifest_path:
                raise ValueError("dataset.manifest_path is required for storage_mode=micro_shards")
            if self.shard_samples is None:
                raise ValueError("dataset.shard_samples is required for storage_mode=micro_shards")
        return self


class OptimizationConfig(BaseModel):
    epochs: int = Field(default=10, ge=1)
    local_steps: int = Field(default=50, ge=1)
    batch_size: int = Field(default=64, ge=1)
    lr: float = Field(default=0.05, gt=0)
    momentum: float = Field(default=0.9, ge=0, le=1)
    weight_decay: float = Field(default=5e-4, ge=0)
    eval_every_epochs: int = Field(default=1, ge=1)


class SyncConfig(BaseModel):
    enabled: bool = True
    averaging: Literal["weights", "deltas"] = "deltas"
    barrier_timeout_s: int = Field(default=120, ge=1)


class AsyncConfig(BaseModel):
    enabled: bool = False
    push_interval_steps: int = Field(default=25, ge=1)
    push_fanout: int = Field(default=2, ge=0)
    max_staleness: int = Field(default=4, ge=0)
    mixing_alpha: float = Field(default=0.5, gt=0, le=1)


class LoggingConfig(BaseModel):
    log_dir: str = "./artifacts/logs"
    checkpoint_dir: str = "./artifacts/checkpoints"
    save_every_round: int = Field(default=1, ge=1)
    metrics_format: Literal["csv", "json"] = "csv"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    seed: int = 42
    algorithm: Literal["local_sgd", "gossip"] = "local_sgd"
    mode: Literal["sync", "async", "hybrid"] = "sync"
    device_preference: list[AcceleratorName] = Field(
        default_factory=lambda: ["cuda", "mps", "cpu"]
    )
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    async_config: AsyncConfig = Field(default_factory=AsyncConfig, alias="async")
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("device_preference")
    @classmethod
    def validate_device_preference(cls, value: list[AcceleratorName]) -> list[AcceleratorName]:
        if len(value) != len(set(value)):
            raise ValueError("device_preference values must be unique")
        return value

    @model_validator(mode="after")
    def validate_mode_toggles(self) -> "TrainingConfig":
        if self.mode == "sync" and not self.sync.enabled:
            raise ValueError("sync mode requires sync.enabled=true")
        if self.mode == "async" and not self.async_config.enabled:
            raise ValueError("async mode requires async.enabled=true")
        if self.mode == "hybrid" and not (self.sync.enabled and self.async_config.enabled):
            raise ValueError("hybrid mode requires both sync.enabled and async.enabled")
        return self


class ResolvedConfig(BaseModel):
    cluster: ClusterConfig
    training: TrainingConfig
    self_node_id: str

    @model_validator(mode="after")
    def validate_self_node(self) -> "ResolvedConfig":
        self.cluster.get_node(self.self_node_id)
        return self

    @property
    def self_node(self) -> NodeConfig:
        return self.cluster.get_node(self.self_node_id)

    def to_display_dict(self) -> dict:
        return {
            "self_node_id": self.self_node_id,
            "self_node": self.self_node.model_dump(),
            "cluster": self.cluster.model_dump(by_alias=True),
            "training": self.training.model_dump(by_alias=True),
        }

    def describe_node(self) -> dict:
        node = self.self_node
        neighbors = [self.cluster.get_node(node_id) for node_id in node.neighbors]
        return {
            "self_node": node.model_dump(),
            "neighbors": [neighbor.model_dump() for neighbor in neighbors],
            "peer_count": len(neighbors),
            "transport": self.cluster.transport,
            "overlay_network": self.cluster.overlay_network,
            "mode": self.training.mode,
            "algorithm": self.training.algorithm,
        }
