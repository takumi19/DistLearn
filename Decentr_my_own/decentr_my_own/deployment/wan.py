from __future__ import annotations

import ipaddress
import socket
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
) -> dict:
    command_name = _command_name(training.mode)
    cluster_arg = _display_path(cluster_path)
    training_arg = _display_path(training_path)
    rounds = training.optimization.epochs
    start_order = _build_start_order(cluster)

    per_node = []
    for node in cluster.nodes:
        cli_prefix = _cli_prefix(node.platform)
        command = (
            f"{cli_prefix} {command_name}"
            f" --cluster {cluster_arg}"
            f" --training {training_arg}"
            f" --self-node {node.id}"
            f" --rounds {rounds}"
            f" --bind-host {node.bind_host}"
        )
        probe_command = (
            f"{cli_prefix} probe-neighbors"
            f" --cluster {cluster_arg}"
            f" --training {training_arg}"
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

    return {
        "cluster_name": cluster.cluster_name,
        "mode": training.mode,
        "overlay_network": cluster.overlay_network,
        "bootstrap_node_id": cluster.bootstrap_node_id,
        "recommended_start_order": start_order,
        "per_node": per_node,
        "shared_steps": [
            "Install Tailscale on every machine and join the same tailnet.",
            "Verify each node.host resolves through MagicDNS or replace it with a 100.x Tailscale IP.",
            "Run commands from the repository root so relative paths stay valid.",
            "Allow inbound TCP on the configured gRPC port on each OS firewall.",
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
