from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import yaml

from decentr_my_own.config.models import ClusterConfig, ResolvedConfig, TrainingConfig


def _expand_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_expand_value(inner) for inner in value]
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def _validate_mapping_payload(payload: Any, source_name: str) -> dict[str, Any]:
    if payload is None:
        raise ValueError(f"Config is empty: {source_name}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML mapping: {source_name}")
    return _expand_value(payload)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return _validate_mapping_payload(payload, str(config_path))


def load_inline_yaml(encoded_payload: str, *, source_name: str) -> dict[str, Any]:
    try:
        raw_bytes = base64.urlsafe_b64decode(encoded_payload.encode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid base64 config payload for {source_name}") from exc
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Inline config payload for {source_name} is not valid UTF-8") from exc
    payload = yaml.safe_load(raw_text)
    return _validate_mapping_payload(payload, source_name)


def load_inventory(path: str | Path) -> tuple["ClusterConfig", "TrainingConfig"]:
    """Load a simplified inventory YAML that contains cluster nodes and training overrides.

    Inventory format example::

        cluster:
          name: demo-cluster
          overlay: tailscale
          bootstrap_node: mac
        nodes:
          mac:
            host: 100.64.0.1
            port: 9090
            bind_host: 0.0.0.0
            platform: macos
            resources: {cpu_cores: 8, relative_speed: 1.0}
            weight: 1.0
          vps1:
            host: 100.64.0.2
            port: 9091
            bind_host: 0.0.0.0
            platform: linux
            resources: {cpu_cores: 2, relative_speed: 0.3}
            weight: 0.3
        training:
          mode: async
          epochs: 50
          batch_size: 64
          lr: 0.05
          scheduler_mode: adaptive
          storage_mode: replicated

    Cluster nodes get a full-mesh neighbor graph automatically.
    Missing training fields use defaults from TrainingConfig.
    """
    payload = load_yaml(path)
    cluster_section = payload.get("cluster", {})
    nodes_section = payload.get("nodes", {})
    training_section = payload.get("training", {})

    if not nodes_section:
        raise ValueError(f"inventory.yaml must have a 'nodes' section: {path}")

    node_ids = list(nodes_section.keys())
    nodes_payload = []
    for node_id, node_data in nodes_section.items():
        node_data = dict(node_data or {})
        node_data["id"] = node_id
        node_data.setdefault("host", "127.0.0.1")
        node_data.setdefault("bind_host", "0.0.0.0")
        node_data.setdefault("port", 50051)
        node_data.setdefault("platform", "unknown")
        node_data.setdefault("weight", 1.0)
        node_data["neighbors"] = [nid for nid in node_ids if nid != node_id]
        resources = node_data.pop("resources", {}) or {}
        node_data["resources"] = {
            "cpu_cores": resources.get("cpu_cores", 2),
            "accelerator": resources.get("accelerator", "cpu"),
            "relative_speed": resources.get("relative_speed", 1.0),
        }
        nodes_payload.append(node_data)

    bootstrap_node_id = cluster_section.get("bootstrap_node") or cluster_section.get("bootstrap_node_id") or node_ids[0]
    cluster_dict = {
        "cluster_name": cluster_section.get("name", "inventory-cluster"),
        "transport": "grpc",
        "overlay_network": cluster_section.get("overlay", cluster_section.get("overlay_network", "tailscale")),
        "tls_enabled": cluster_section.get("tls_enabled", False),
        "bootstrap_node_id": bootstrap_node_id,
        "nodes": nodes_payload,
    }
    cluster = ClusterConfig.model_validate(cluster_dict)

    # Build training config: start from defaults, then apply inventory overrides.
    ts = training_section
    dataset_overrides: dict = {}
    for key in ("storage_mode", "scheduler_mode", "manifest_path", "cache_dir",
                "shard_samples", "partitioning", "rebalance_window_batches",
                "warmup_windows", "min_local_shards", "prefetch_shards",
                "max_cache_bytes", "throughput_ema", "fake_train_size",
                "fake_val_size", "fake_test_size", "num_workers"):
        if key in ts:
            dataset_overrides[key] = ts[key]

    opt_overrides: dict = {}
    for key in ("epochs", "batch_size", "lr", "momentum", "weight_decay",
                "eval_every_epochs", "local_steps"):
        if key in ts:
            opt_overrides[key] = ts[key]

    training_dict: dict = {}
    if "mode" in ts:
        training_dict["mode"] = ts["mode"]
    if "algorithm" in ts:
        training_dict["algorithm"] = ts["algorithm"]
    if "seed" in ts:
        training_dict["seed"] = ts["seed"]
    if "model" in ts:
        training_dict["model"] = ts["model"]
    if "dataset" in ts and isinstance(ts["dataset"], dict):
        training_dict["dataset"] = ts["dataset"]
    elif dataset_overrides:
        training_dict["dataset"] = dataset_overrides

    if opt_overrides:
        training_dict["optimization"] = opt_overrides

    if "async" in ts:
        training_dict["async"] = ts["async"]
    if "sync" in ts:
        training_dict["sync"] = ts["sync"]
    if "logging" in ts:
        training_dict["logging"] = ts["logging"]

    # Ensure mode toggles are consistent: async mode needs async.enabled=true.
    mode = training_dict.get("mode", "async")
    if mode == "async":
        if "async" not in training_dict:
            training_dict["async"] = {}
        training_dict["async"]["enabled"] = True
        if "sync" not in training_dict:
            training_dict["sync"] = {"enabled": False}
    elif mode == "sync":
        if "sync" not in training_dict:
            training_dict["sync"] = {}
        training_dict["sync"]["enabled"] = True
        if "async" not in training_dict:
            training_dict["async"] = {"enabled": False}

    training = TrainingConfig.model_validate(training_dict)
    return cluster, training


def load_cluster_config(path: str | Path) -> ClusterConfig:
    return ClusterConfig.model_validate(load_yaml(path))


def load_cluster_config_inline(encoded_payload: str) -> ClusterConfig:
    return ClusterConfig.model_validate(
        load_inline_yaml(encoded_payload, source_name="--cluster-b64")
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.model_validate(load_yaml(path))


def load_training_config_inline(encoded_payload: str) -> TrainingConfig:
    return TrainingConfig.model_validate(
        load_inline_yaml(encoded_payload, source_name="--training-b64")
    )


def load_resolved_config(
    cluster_path: str | Path, training_path: str | Path, self_node_id: str
) -> ResolvedConfig:
    return ResolvedConfig(
        cluster=load_cluster_config(cluster_path),
        training=load_training_config(training_path),
        self_node_id=self_node_id,
    )


def load_resolved_config_inline(
    cluster_b64: str, training_b64: str, self_node_id: str
) -> ResolvedConfig:
    return ResolvedConfig(
        cluster=load_cluster_config_inline(cluster_b64),
        training=load_training_config_inline(training_b64),
        self_node_id=self_node_id,
    )
