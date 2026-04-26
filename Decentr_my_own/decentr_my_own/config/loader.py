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
