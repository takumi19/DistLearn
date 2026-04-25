from __future__ import annotations

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


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return _expand_value(payload)


def load_cluster_config(path: str | Path) -> ClusterConfig:
    return ClusterConfig.model_validate(load_yaml(path))


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.model_validate(load_yaml(path))


def load_resolved_config(
    cluster_path: str | Path, training_path: str | Path, self_node_id: str
) -> ResolvedConfig:
    return ResolvedConfig(
        cluster=load_cluster_config(cluster_path),
        training=load_training_config(training_path),
        self_node_id=self_node_id,
    )
