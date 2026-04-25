from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from decentr_my_own.config.loader import load_cluster_config, load_resolved_config, load_training_config
from decentr_my_own.data.loaders import build_partition_summary
from decentr_my_own.data.shards import build_dataset_shards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decentr-my-own",
        description="Inspect and validate the decentralized training configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate cluster and training YAML files for a given node.",
    )
    _add_resolved_args(validate_parser)

    show_parser = subparsers.add_parser(
        "show-config",
        help="Print the resolved runtime config for a given node as JSON.",
    )
    _add_resolved_args(show_parser)

    describe_parser = subparsers.add_parser(
        "describe-node",
        help="Print the current node and its neighbors as JSON.",
    )
    describe_parser.add_argument("--cluster", type=Path, required=True)
    describe_parser.add_argument("--training", type=Path, required=True)
    describe_parser.add_argument("--self-node", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-files",
        help="Load cluster/training files separately and print a short summary.",
    )
    inspect_parser.add_argument("--cluster", type=Path, required=True)
    inspect_parser.add_argument("--training", type=Path, required=True)

    shard_parser = subparsers.add_parser(
        "build-shards",
        help="Build deterministic local micro-shards and a manifest from the training config.",
    )
    shard_parser.add_argument("--training", type=Path, required=True)
    shard_parser.add_argument("--force", action="store_true")

    local_train_parser = subparsers.add_parser(
        "local-train",
        help="Run local single-process training using the shared training pipeline.",
    )
    _add_resolved_args(local_train_parser)
    local_train_parser.add_argument("--run-name")
    local_train_parser.add_argument("--epochs", type=int)
    local_train_parser.add_argument("--max-train-batches", type=int)
    local_train_parser.add_argument("--max-eval-batches", type=int)

    partition_parser = subparsers.add_parser(
        "inspect-partition",
        help="Print the deterministic train partition for the selected node.",
    )
    _add_resolved_args(partition_parser)
    partition_parser.add_argument("--shuffle-token", type=int, default=0)

    ping_parser = subparsers.add_parser(
        "ping-remote",
        help="Ping a remote peer transport endpoint.",
    )
    ping_parser.add_argument("--target", required=True, help="Remote address host:port")
    ping_parser.add_argument("--sender-node", required=True)

    remote_state_parser = subparsers.add_parser(
        "remote-state",
        help="Fetch stored payload summaries from a remote peer transport endpoint.",
    )
    remote_state_parser.add_argument("--target", required=True, help="Remote address host:port")

    smoke_parser = subparsers.add_parser(
        "peer-smoke",
        help="Run a local multi-process gRPC transport smoke test.",
    )
    smoke_parser.add_argument("--peer-count", type=int, default=3)

    serve_parser = subparsers.add_parser(
        "serve-peer",
        help="Start a local peer transport server for the selected node.",
    )
    _add_resolved_args(serve_parser)
    serve_parser.add_argument("--bind-host")
    serve_parser.add_argument("--bind-port", type=int)
    serve_parser.add_argument("--shutdown-after-s", type=float)

    sync_smoke_parser = subparsers.add_parser(
        "sync-smoke",
        help="Run a local multi-process synchronous training smoke test.",
    )
    sync_smoke_parser.add_argument("--peer-count", type=int, default=3)
    sync_smoke_parser.add_argument("--rounds", type=int, default=1)

    run_sync_parser = subparsers.add_parser(
        "run-sync-node",
        help="Run one synchronous training node using the configured peer graph.",
    )
    _add_resolved_args(run_sync_parser)
    run_sync_parser.add_argument("--rounds", type=int, default=1)
    run_sync_parser.add_argument("--max-local-batches", type=int)
    run_sync_parser.add_argument("--max-eval-batches", type=int)
    run_sync_parser.add_argument("--run-name")
    run_sync_parser.add_argument("--transport-timeout-s", type=float, default=15.0)
    run_sync_parser.add_argument("--bind-host")

    async_smoke_parser = subparsers.add_parser(
        "async-smoke",
        help="Run a local multi-process asynchronous training smoke test.",
    )
    async_smoke_parser.add_argument("--peer-count", type=int, default=3)
    async_smoke_parser.add_argument("--rounds", type=int, default=2)

    run_async_parser = subparsers.add_parser(
        "run-async-node",
        help="Run one asynchronous training node using the configured peer graph.",
    )
    _add_resolved_args(run_async_parser)
    run_async_parser.add_argument("--rounds", type=int, default=2)
    run_async_parser.add_argument("--max-local-batches", type=int)
    run_async_parser.add_argument("--max-eval-batches", type=int)
    run_async_parser.add_argument("--run-name")
    run_async_parser.add_argument("--transport-timeout-s", type=float, default=15.0)
    run_async_parser.add_argument("--bind-host")
    run_async_parser.add_argument("--round-delay-s", type=float, default=0.0)
    run_async_parser.add_argument("--shutdown-grace-s", type=float, default=2.0)

    wan_preflight_parser = subparsers.add_parser(
        "wan-preflight",
        help="Validate that the selected node config is ready for WAN/Tailscale launch.",
    )
    _add_resolved_args(wan_preflight_parser)
    wan_preflight_parser.add_argument("--check-dns", action="store_true")

    launch_plan_parser = subparsers.add_parser(
        "launch-plan",
        help="Print per-node launch and probe commands for the current cluster config.",
    )
    launch_plan_parser.add_argument("--cluster", type=Path, required=True)
    launch_plan_parser.add_argument("--training", type=Path, required=True)

    probe_parser = subparsers.add_parser(
        "probe-neighbors",
        help="Ping every configured neighbor of the selected node.",
    )
    _add_resolved_args(probe_parser)
    probe_parser.add_argument("--timeout-s", type=float, default=3.0)
    probe_parser.add_argument("--include-state", action="store_true")

    report_parser = subparsers.add_parser(
        "report-run",
        help="Aggregate all node summaries for one run into a single report.",
    )
    report_parser.add_argument("--log-root", type=Path, required=True)
    report_parser.add_argument("--run-id", required=True)

    compare_parser = subparsers.add_parser(
        "compare-runs",
        help="Compare two aggregated run reports under the same log root.",
    )
    compare_parser.add_argument("--log-root", type=Path, required=True)
    compare_parser.add_argument("--baseline-run", required=True)
    compare_parser.add_argument("--candidate-run", required=True)

    compare_smoke_parser = subparsers.add_parser(
        "compare-smoke",
        help="Run sync and async smoke experiments and print a comparison report.",
    )
    compare_smoke_parser.add_argument("--peer-count", type=int, default=3)
    compare_smoke_parser.add_argument("--sync-rounds", type=int, default=1)
    compare_smoke_parser.add_argument("--async-rounds", type=int, default=2)

    return parser


def _add_resolved_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cluster", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--self-node", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        resolved = load_resolved_config(args.cluster, args.training, args.self_node)
        print(
            "Configuration is valid for "
            f"node '{resolved.self_node_id}' in cluster '{resolved.cluster.cluster_name}'."
        )
        return 0

    if args.command == "show-config":
        resolved = load_resolved_config(args.cluster, args.training, args.self_node)
        print(json.dumps(resolved.to_display_dict(), indent=2))
        return 0

    if args.command == "describe-node":
        resolved = load_resolved_config(args.cluster, args.training, args.self_node)
        print(json.dumps(resolved.describe_node(), indent=2))
        return 0

    if args.command == "inspect-files":
        cluster = load_cluster_config(args.cluster)
        training = load_training_config(args.training)
        summary = {
            "cluster_name": cluster.cluster_name,
            "node_count": len(cluster.nodes),
            "mode": training.mode,
            "algorithm": training.algorithm,
            "dataset": training.dataset.name,
            "model": training.model.name,
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "build-shards":
        training = load_training_config(args.training)
        print(json.dumps(build_dataset_shards(training, force=args.force).to_dict(), indent=2))
        return 0

    if args.command == "local-train":
        from decentr_my_own.training.engine import LocalTrainOverrides, run_local_training

        resolved = load_resolved_config(args.cluster, args.training, args.self_node)
        result = run_local_training(
            resolved,
            LocalTrainOverrides(
                epochs=args.epochs,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                run_name=args.run_name,
            ),
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "inspect-partition":
        resolved = load_resolved_config(args.cluster, args.training, args.self_node)
        print(json.dumps(build_partition_summary(resolved, args.shuffle_token), indent=2))
        return 0

    if args.command == "ping-remote":
        from decentr_my_own.comm.client import PeerClient

        client = PeerClient(args.target)
        try:
            print(json.dumps(client.ping(args.sender_node), indent=2))
        finally:
            client.close()
        return 0

    if args.command == "remote-state":
        from decentr_my_own.comm.client import PeerClient

        client = PeerClient(args.target)
        try:
            print(json.dumps(client.get_peer_state(), indent=2))
        finally:
            client.close()
        return 0

    if args.command == "peer-smoke":
        from decentr_my_own.comm.smoke import run_peer_smoke

        print(json.dumps(run_peer_smoke(peer_count=args.peer_count), indent=2))
        return 0

    if args.command == "serve-peer":
        from decentr_my_own.comm.server import PeerServer

        resolved = load_resolved_config(args.cluster, args.training, args.self_node)
        port = args.bind_port or resolved.self_node.port
        server = PeerServer(
            node_id=resolved.self_node_id,
            host=args.bind_host or resolved.self_node.bind_host,
            port=port,
        )
        server.start()
        print(
            json.dumps(
                {
                    "node_id": resolved.self_node_id,
                    "bind_address": server.address,
                    "advertise_address": f"{resolved.self_node.host}:{port}",
                    "mode": resolved.training.mode,
                },
                indent=2,
            )
        )
        try:
            if args.shutdown_after_s is not None:
                time.sleep(max(args.shutdown_after_s, 0.0))
            else:
                server.wait_for_termination()
        except KeyboardInterrupt:
            pass
        finally:
            server.stop(grace=0.0)
        return 0

    if args.command == "sync-smoke":
        from decentr_my_own.algorithms.sync_barrier import run_sync_smoke

        print(
            json.dumps(
                run_sync_smoke(peer_count=args.peer_count, rounds=args.rounds),
                indent=2,
            )
        )
        return 0

    if args.command == "run-sync-node":
        from decentr_my_own.algorithms.sync_barrier import SyncRunOverrides, run_sync_worker

        resolved = load_resolved_config(args.cluster, args.training, args.self_node)
        print(
            json.dumps(
                run_sync_worker(
                    resolved,
                    overrides=SyncRunOverrides(
                        rounds=args.rounds,
                        max_local_batches=args.max_local_batches,
                        max_eval_batches=args.max_eval_batches,
                        run_name=args.run_name,
                        transport_timeout_s=args.transport_timeout_s,
                        bind_host=args.bind_host,
                    ),
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "async-smoke":
        from decentr_my_own.algorithms.async_gossip import run_async_smoke

        print(
            json.dumps(
                run_async_smoke(peer_count=args.peer_count, rounds=args.rounds),
                indent=2,
            )
        )
        return 0

    if args.command == "run-async-node":
        from decentr_my_own.algorithms.async_gossip import AsyncRunOverrides, run_async_worker

        resolved = load_resolved_config(args.cluster, args.training, args.self_node)
        print(
            json.dumps(
                run_async_worker(
                    resolved,
                    overrides=AsyncRunOverrides(
                        rounds=args.rounds,
                        max_local_batches=args.max_local_batches,
                        max_eval_batches=args.max_eval_batches,
                        run_name=args.run_name,
                        transport_timeout_s=args.transport_timeout_s,
                        bind_host=args.bind_host,
                        round_delay_s=args.round_delay_s,
                        shutdown_grace_s=args.shutdown_grace_s,
                    ),
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "wan-preflight":
        from decentr_my_own.deployment.wan import build_wan_preflight

        resolved = load_resolved_config(args.cluster, args.training, args.self_node)
        print(json.dumps(build_wan_preflight(resolved, check_dns=args.check_dns), indent=2))
        return 0

    if args.command == "launch-plan":
        from decentr_my_own.deployment.wan import build_launch_plan

        cluster = load_cluster_config(args.cluster)
        training = load_training_config(args.training)
        print(
            json.dumps(
                build_launch_plan(
                    cluster_path=args.cluster,
                    training_path=args.training,
                    cluster=cluster,
                    training=training,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "probe-neighbors":
        from decentr_my_own.deployment.wan import probe_neighbors

        resolved = load_resolved_config(args.cluster, args.training, args.self_node)
        print(
            json.dumps(
                probe_neighbors(
                    resolved,
                    timeout_s=args.timeout_s,
                    include_state=args.include_state,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "report-run":
        from decentr_my_own.metrics.reporting import build_run_report

        print(json.dumps(build_run_report(args.log_root, args.run_id), indent=2))
        return 0

    if args.command == "compare-runs":
        from decentr_my_own.metrics.reporting import build_run_report, compare_run_reports

        baseline = build_run_report(args.log_root, args.baseline_run)
        candidate = build_run_report(args.log_root, args.candidate_run)
        print(json.dumps(compare_run_reports(baseline, candidate), indent=2))
        return 0

    if args.command == "compare-smoke":
        from decentr_my_own.metrics.reporting import build_smoke_comparison

        print(
            json.dumps(
                build_smoke_comparison(
                    peer_count=args.peer_count,
                    sync_rounds=args.sync_rounds,
                    async_rounds=args.async_rounds,
                ),
                indent=2,
            )
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
