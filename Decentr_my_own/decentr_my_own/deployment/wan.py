from __future__ import annotations

import base64
import ipaddress
import json
import socket
from datetime import datetime
from pathlib import Path

from decentr_my_own.config.models import ClusterConfig, NodeConfig, ResolvedConfig, TrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAILSCALE_RANGE = ipaddress.ip_network("100.64.0.0/10")


def build_wan_preflight(
    resolved: ResolvedConfig,
    *,
    check_dns: bool = False,
) -> dict:
    cluster = resolved.cluster
    start_order = _build_start_order(cluster)
    notes = []
    warnings = []

    if cluster.overlay_network == "tailscale":
        notes.append(
            "Use Tailscale MagicDNS names (*.ts.net) or 100.x Tailscale IPs in node.host."
        )
        if not cluster.tls_enabled:
            notes.append(
                "tls_enabled=false is acceptable for v1 when traffic stays inside the Tailscale tailnet."
            )

    if resolved.self_node.bind_host in {"127.0.0.1", "localhost"} and len(cluster.nodes) > 1:
        warnings.append(
            f"Self node '{resolved.self_node_id}' binds to loopback; remote peers will not reach it."
        )

    node_checks = []
    for node in cluster.nodes:
        resolve_result = _resolve_host(node.host) if check_dns else {"ok": None, "addresses": []}
        node_warnings = _build_node_warnings(node=node, overlay_network=cluster.overlay_network)
        if check_dns and resolve_result["ok"] is False:
            node_warnings.append(
                f"Node '{node.id}' host '{node.host}' did not resolve on this machine."
            )
        warnings.extend(node_warnings)
        node_checks.append(
            {
                "node_id": node.id,
                "platform": node.platform,
                "advertise_target": _node_target(node.host, node.port),
                "bind_target": _node_target(node.bind_host, node.port),
                "is_self": node.id == resolved.self_node_id,
                "is_neighbor": node.id in resolved.self_node.neighbors,
                "dns_checked": check_dns,
                "dns_ok": resolve_result["ok"],
                "resolved_addresses": resolve_result["addresses"],
            }
        )

    return {
        "cluster_name": cluster.cluster_name,
        "self_node_id": resolved.self_node_id,
        "mode": resolved.training.mode,
        "overlay_network": cluster.overlay_network,
        "transport": cluster.transport,
        "tls_enabled": cluster.tls_enabled,
        "bootstrap_node_id": cluster.bootstrap_node_id,
        "recommended_start_order": start_order,
        "self_bind_target": _node_target(resolved.self_node.bind_host, resolved.self_node.port),
        "self_advertise_target": _node_target(resolved.self_node.host, resolved.self_node.port),
        "neighbor_targets": [
            _node_target(cluster.get_node(node_id).host, cluster.get_node(node_id).port)
            for node_id in resolved.self_node.neighbors
        ],
        "firewall_requirements": [
            f"Allow inbound TCP {resolved.self_node.port} on '{resolved.self_node_id}'."
        ],
        "notes": notes,
        "warnings": warnings,
        "nodes": node_checks,
    }


def build_launch_plan(
    *,
    cluster_path: str | Path,
    training_path: str | Path,
    cluster: ClusterConfig,
    training: TrainingConfig,
    run_name: str | None = None,
    inline_configs: bool = False,
    selected_node_ids: list[str] | None = None,
    bootstrap_node_id: str | None = None,
) -> dict:
    launch_cluster = _select_launch_cluster(
        cluster,
        selected_node_ids=selected_node_ids,
        bootstrap_node_id=bootstrap_node_id,
    )
    command_name = _command_name(training.mode)
    cluster_arg = _display_path(cluster_path)
    training_arg = _display_path(training_path)
    epochs = training.optimization.epochs
    start_order = _build_start_order(launch_cluster)
    shared_run_name = run_name or _default_run_name(launch_cluster)
    effective_inline_configs = (
        inline_configs
        or selected_node_ids is not None
        or bootstrap_node_id is not None
    )

    if effective_inline_configs:
        cluster_source_args = f" --cluster-b64 {_encode_inline_config(launch_cluster.model_dump())}"
        encoded_training = _encode_inline_config(training.model_dump(by_alias=True))
        training_source_args = f" --training-b64 {encoded_training}"
        config_distribution = "inline"
    else:
        encoded_training = None
        cluster_source_args = f" --cluster {cluster_arg}"
        training_source_args = f" --training {training_arg}"
        config_distribution = "files"

    per_node = []
    for node in launch_cluster.nodes:
        cli_prefix = _cli_prefix(node.platform)
        command = (
            f"{cli_prefix} {command_name}"
            f"{cluster_source_args}"
            f"{training_source_args}"
            f" --self-node {node.id}"
            f" --epochs {epochs}"
            f" --run-name {shared_run_name}"
            f" --bind-host {node.bind_host}"
        )
        probe_command = (
            f"{cli_prefix} probe-neighbors"
            f"{cluster_source_args}"
            f"{training_source_args}"
            f" --self-node {node.id}"
            " --include-state"
        )
        per_node.append(
            {
                "node_id": node.id,
                "platform": node.platform,
                "advertise_target": _node_target(node.host, node.port),
                "bind_target": _node_target(node.bind_host, node.port),
                "run_command": command,
                "probe_command": probe_command,
            }
        )

    bootstrap_prepare_command = None
    if training.dataset.storage_mode == "micro_shards":
        prepare_node_id = launch_cluster.bootstrap_node_id or launch_cluster.nodes[0].id
        bootstrap_node = launch_cluster.get_node(prepare_node_id)
        cli_prefix = _cli_prefix(bootstrap_node.platform)
        if effective_inline_configs and encoded_training is not None:
            bootstrap_prepare_command = (
                f"{cli_prefix} build-shards"
                f" --training-b64 {encoded_training}"
                " --force"
            )
        else:
            bootstrap_prepare_command = (
                f"{cli_prefix} build-shards"
                f" --training {training_arg}"
                " --force"
            )

    return {
        "cluster_name": launch_cluster.cluster_name,
        "mode": training.mode,
        "overlay_network": launch_cluster.overlay_network,
        "bootstrap_node_id": launch_cluster.bootstrap_node_id,
        "shared_run_name": shared_run_name,
        "config_distribution": config_distribution,
        "storage_mode": training.dataset.storage_mode,
        "scheduler_mode": training.dataset.scheduler_mode,
        "selected_node_ids": [node.id for node in launch_cluster.nodes],
        "bootstrap_prepare_command": bootstrap_prepare_command,
        "recommended_start_order": start_order,
        "per_node": per_node,
        "shared_steps": [
            "Install Tailscale on every machine and join the same tailnet.",
            "Verify each node.host resolves through MagicDNS or replace it with a 100.x Tailscale IP.",
            (
                "Run commands from the repository root so relative paths stay valid."
                if not effective_inline_configs
                else "Inline launch commands already include the full cluster and training config."
            ),
            "Allow inbound TCP on the configured gRPC port on each OS firewall.",
            (
                "Run bootstrap_prepare_command on the bootstrap node before training when storage_mode=micro_shards."
                if training.dataset.storage_mode == "micro_shards"
                else "Replicated mode does not require shard prebuild."
            ),
        ],
    }


def probe_neighbors(
    resolved: ResolvedConfig,
    *,
    timeout_s: float = 3.0,
    include_state: bool = False,
) -> dict:
    from decentr_my_own.comm.client import PeerClient

    cluster = resolved.cluster
    results = []
    reachable_count = 0
    for neighbor_id in resolved.self_node.neighbors:
        neighbor = cluster.get_node(neighbor_id)
        client = PeerClient(_node_target(neighbor.host, neighbor.port))
        try:
            ping_result = client.ping(sender_node_id=resolved.self_node_id, timeout_s=timeout_s)
            result = {
                "neighbor_id": neighbor.id,
                "target": _node_target(neighbor.host, neighbor.port),
                "reachable": True,
                "ping": ping_result,
            }
            if include_state:
                result["remote_state"] = client.get_peer_state(timeout_s=timeout_s)
            reachable_count += 1
        except Exception as exc:
            result = {
                "neighbor_id": neighbor.id,
                "target": _node_target(neighbor.host, neighbor.port),
                "reachable": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            client.close()
        results.append(result)

    return {
        "cluster_name": cluster.cluster_name,
        "self_node_id": resolved.self_node_id,
        "mode": resolved.training.mode,
        "reachable_count": reachable_count,
        "neighbor_count": len(results),
        "results": results,
    }


def _build_start_order(cluster: ClusterConfig) -> list[str]:
    ordered = [node.id for node in cluster.nodes]
    if cluster.bootstrap_node_id is None or cluster.bootstrap_node_id not in ordered:
        return ordered
    return [cluster.bootstrap_node_id] + [
        node_id for node_id in ordered if node_id != cluster.bootstrap_node_id
    ]


def _select_launch_cluster(
    cluster: ClusterConfig,
    *,
    selected_node_ids: list[str] | None,
    bootstrap_node_id: str | None,
) -> ClusterConfig:
    if selected_node_ids is None and bootstrap_node_id is None:
        return cluster

    if selected_node_ids is None:
        payload = cluster.model_dump()
        payload["bootstrap_node_id"] = bootstrap_node_id
        return ClusterConfig(**payload)

    if not selected_node_ids:
        raise ValueError("--nodes must include at least one node id")

    seen: set[str] = set()
    ordered_node_ids: list[str] = []
    for node_id in selected_node_ids:
        if node_id in seen:
            continue
        cluster.get_node(node_id)
        seen.add(node_id)
        ordered_node_ids.append(node_id)

    selected_set = set(ordered_node_ids)
    effective_bootstrap = bootstrap_node_id or cluster.bootstrap_node_id or ordered_node_ids[0]
    if effective_bootstrap not in selected_set:
        raise ValueError(
            f"bootstrap node '{effective_bootstrap}' must be present in the selected --nodes set"
        )

    nodes_payload = []
    for node_id in ordered_node_ids:
        node_payload = cluster.get_node(node_id).model_dump()
        node_payload["neighbors"] = [
            neighbor_id for neighbor_id in ordered_node_ids if neighbor_id != node_id
        ]
        nodes_payload.append(node_payload)

    return ClusterConfig(
        cluster_name=cluster.cluster_name,
        transport=cluster.transport,
        overlay_network=cluster.overlay_network,
        tls_enabled=cluster.tls_enabled,
        bootstrap_node_id=effective_bootstrap,
        nodes=[NodeConfig(**node_payload) for node_payload in nodes_payload],
    )


def _build_node_warnings(*, node: NodeConfig, overlay_network: str) -> list[str]:
    warnings = []
    if overlay_network == "tailscale":
        if _is_loopback_host(node.host):
            warnings.append(
                f"Node '{node.id}' host '{node.host}' is loopback; use MagicDNS or a 100.x Tailscale IP."
            )
        elif not _looks_like_tailscale_host(node.host):
            warnings.append(
                f"Node '{node.id}' host '{node.host}' does not look like a Tailscale address."
            )
        if node.bind_host in {"127.0.0.1", "localhost"}:
            warnings.append(
                f"Node '{node.id}' bind_host is loopback; set bind_host to 0.0.0.0 for multi-host runs."
            )
    return warnings


def _resolve_host(host: str) -> dict:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return {"ok": False, "addresses": []}
    addresses = sorted({info[4][0] for info in infos if info[4]})
    return {"ok": True, "addresses": addresses}


def _looks_like_tailscale_host(host: str) -> bool:
    if host.endswith(".ts.net"):
        return True
    try:
        return ipaddress.ip_address(host) in TAILSCALE_RANGE
    except ValueError:
        return False


def _is_loopback_host(host: str) -> bool:
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _display_path(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path_obj.as_posix()


def _default_run_name(cluster: ClusterConfig) -> str:
    return f"{cluster.cluster_name}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"


def _encode_inline_config(payload: dict) -> str:
    raw_text = json.dumps(payload, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw_text.encode("utf-8")).decode("utf-8")


def _command_name(mode: str) -> str:
    if mode == "sync":
        return "run-sync-node"
    if mode == "async":
        return "run-async-node"
    raise ValueError("launch plan currently supports sync and async modes only")


def _cli_prefix(platform: str) -> str:
    if platform == "windows":
        return "$env:PYTHONPATH='Decentr_my_own'; python -m decentr_my_own.cli"
    return "PYTHONPATH=Decentr_my_own python3 -m decentr_my_own.cli"


def _node_target(host: str, port: int) -> str:
    return f"{host}:{port}"
