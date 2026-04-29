from __future__ import annotations

import multiprocessing as mp
import socket
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from decentr_my_own.comm.client import PeerClient
from decentr_my_own.comm.messages import PayloadMetadata, PeerPayload
from decentr_my_own.comm.server import PeerServer
from decentr_my_own.config.loader import load_resolved_config, load_yaml
from decentr_my_own.config.models import ResolvedConfig
from decentr_my_own.data.runtime import AdaptiveMicroShardRuntime, prepare_static_micro_shards
from decentr_my_own.data.loaders import build_local_dataloaders
from decentr_my_own.data.manifest import load_manifest, save_manifest
from decentr_my_own.data.scheduler_state import RunCompletionRecord
from decentr_my_own.data.shards import build_dataset_shards
from decentr_my_own.models.factory import build_model
from decentr_my_own.training.device import select_device
from decentr_my_own.training.io import (
    create_run_directories,
    safe_rate,
    utc_now_iso,
    write_metrics,
    write_summary,
)
from decentr_my_own.training.metrics import (
    init_confusion_matrix,
    macro_f1_from_confusion,
    summarize_classification_metrics,
    update_confusion_matrix,
)
from decentr_my_own.training.seed import set_global_seed
from decentr_my_own.training.state_ops import (
    check_state_finite,
    digest_state,
    extract_model_state,
    load_model_state,
    require_state_finite,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAX_PUSH_ATTEMPT_TIMEOUT_S = 30.0


@dataclass
class AsyncRunOverrides:
    epochs: int | None = None
    rounds: int | None = None
    max_local_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str | None = None
    transport_timeout_s: float = 15.0
    bind_host: str | None = None
    round_delay_s: float = 0.0
    shutdown_grace_s: float = 2.0
    serve_config: bool = False  # start HTTP config server so followers can join without a token


def run_async_worker(
    resolved: ResolvedConfig, overrides: AsyncRunOverrides | None = None
) -> dict:
    overrides = overrides or AsyncRunOverrides()
    config = resolved.training
    epoch_count = _resolve_async_epoch_count(config, overrides)
    started_at = utc_now_iso()
    run_started_perf = time.perf_counter()
    set_global_seed(config.seed)
    device = select_device([str(item) for item in config.device_preference])
    pin_memory = device.type == "cuda"

    run_id, log_dir, checkpoint_dir = create_run_directories(
        config.logging.log_dir,
        config.logging.checkpoint_dir,
        resolved.self_node_id,
        overrides.run_name,
    )
    model = build_model(config.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.optimization.lr,
        momentum=config.optimization.momentum,
        weight_decay=config.optimization.weight_decay,
    )

    server = PeerServer(
        node_id=resolved.self_node_id,
        host=overrides.bind_host or resolved.self_node.bind_host,
        port=resolved.self_node.port,
        shard_manifest_path=(
            config.dataset.manifest_path
            if config.dataset.storage_mode == "micro_shards"
            and config.dataset.manifest_path is not None
            and Path(config.dataset.manifest_path).exists()
            else None
        ),
        transfer_chunk_bytes=config.dataset.transfer_chunk_bytes,
    )
    server.start()
    neighbors = [resolved.cluster.get_node(node_id) for node_id in resolved.self_node.neighbors]
    adaptive_runtime = None
    static_shard_ids: list[str] | None = None

    config_http_server = None
    if overrides.serve_config:
        from decentr_my_own.comm.config_server import RunConfigServer, config_port_for
        _bootstrap_nid = resolved.cluster.bootstrap_node_id or resolved.self_node_id
        if resolved.self_node_id == _bootstrap_nid:
            _cfg_payload = {
                "cluster": resolved.cluster.model_dump(),
                "training": resolved.training.model_dump(by_alias=True),
                "run_name": overrides.run_name,
                "epochs": epoch_count,
            }
            config_http_server = RunConfigServer(
                host=overrides.bind_host or resolved.self_node.bind_host,
                port=config_port_for(resolved.self_node.port),
                payload=_cfg_payload,
            )
            config_http_server.start()

    try:
        _wait_for_neighbors(
            self_node_id=resolved.self_node_id,
            neighbors=neighbors,
            timeout_s=overrides.transport_timeout_s,
        )
        if (
            config.dataset.storage_mode == "micro_shards"
            and config.dataset.scheduler_mode == "adaptive"
        ):
            adaptive_runtime = AdaptiveMicroShardRuntime(
                resolved=resolved,
                server=server,
                timeout_s=overrides.transport_timeout_s,
            )
        else:
            static_shard_ids = prepare_static_micro_shards(
                resolved,
                server,
                timeout_s=overrides.transport_timeout_s,
                window_id=0,
            )
        dataloaders = None

        epoch_history = []
        mixed_peer_updates_total = 0
        dropped_peer_updates_total = 0
        nonfinite_peer_updates_total = 0
        max_observed_staleness = 0
        failed_pushes_total = 0
        push_count_total = 0
        local_step = 0
        last_mixed_payload_ids: dict[str, str] = {}
        final_test_metrics = {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}
        next_adaptive_window_id = 0

        for epoch_idx in range(epoch_count):
            epoch_started_perf = time.perf_counter()
            epoch_train_accumulator = _init_train_accumulator(config.model.num_classes)
            epoch_mixed_peer_updates = 0
            epoch_failed_pushes = 0
            epoch_push_count = 0
            epoch_max_staleness = 0
            epoch_mixed_senders: set[str] = set()
            epoch_dropped_peer_updates = 0
            epoch_nonfinite_peer_updates = 0
            epoch_dropped_peer_senders: set[str] = set()
            epoch_drop_reasons: list[str] = []
            epoch_window_count = 0
            epoch_window_rows: list[dict[str, int | float]] = []

            while True:
                window_assignment = None
                if adaptive_runtime is not None:
                    window_assignment = adaptive_runtime.get_window_assignment(next_adaptive_window_id)
                    if window_assignment.epoch_id != epoch_idx:
                        raise RuntimeError(
                            "Adaptive runtime returned mismatched epoch window: "
                            f"expected epoch_id={epoch_idx}, got {window_assignment.epoch_id}"
                        )
                    train_shard_ids = window_assignment.shard_ids
                else:
                    train_shard_ids = static_shard_ids

                dataloaders = build_local_dataloaders(
                    resolved,
                    pin_memory=pin_memory,
                    train_shard_ids=train_shard_ids,
                )
                if dataloaders.train_sampler is not None:
                    dataloaders.train_sampler.set_epoch(epoch_idx)

                train_started_perf = time.perf_counter()
                train_metrics = _train_async_window(
                    model,
                    dataloaders.train,
                    optimizer,
                    criterion,
                    device,
                    num_classes=config.model.num_classes,
                    max_batches=overrides.max_local_batches,
                    start_step=local_step,
                    run_id=run_id,
                    self_node_id=resolved.self_node_id,
                    server=server,
                    neighbors=neighbors,
                    push_fanout=config.async_config.push_fanout,
                    push_interval_steps=config.async_config.push_interval_steps,
                    base_alpha=config.async_config.mixing_alpha,
                    max_staleness=config.async_config.max_staleness,
                    transport_timeout_s=overrides.transport_timeout_s,
                    last_mixed_payload_ids=last_mixed_payload_ids,
                )
                local_step = train_metrics["last_step"]
                if overrides.round_delay_s > 0:
                    time.sleep(overrides.round_delay_s)
                train_duration_s = time.perf_counter() - train_started_perf

                if adaptive_runtime is not None and window_assignment is not None:
                    adaptive_runtime.report_window(
                        window_id=window_assignment.window_id,
                        samples_processed=train_metrics["samples_processed"],
                        duration_s=train_duration_s,
                    )
                    epoch_window_rows.append(
                        {
                            "window_id": window_assignment.window_id,
                            "epoch_window_index": window_assignment.epoch_window_index,
                            "epoch_window_count": window_assignment.epoch_window_count,
                            "assigned_shard_count": len(window_assignment.shard_ids),
                            "samples_processed": train_metrics["samples_processed"],
                            "duration_s": train_duration_s,
                        }
                    )
                    next_adaptive_window_id += 1

                _accumulate_train_metrics(epoch_train_accumulator, train_metrics)
                epoch_mixed_peer_updates += train_metrics["mixed_peer_updates"]
                epoch_failed_pushes += train_metrics["failed_pushes"]
                epoch_push_count += train_metrics["pushes_sent"]
                epoch_max_staleness = max(epoch_max_staleness, train_metrics["max_staleness"])
                epoch_mixed_senders.update(train_metrics["mixed_senders"])
                epoch_dropped_peer_updates += train_metrics["dropped_peer_updates"]
                epoch_nonfinite_peer_updates += train_metrics["nonfinite_peer_updates"]
                epoch_dropped_peer_senders.update(train_metrics["dropped_peer_senders"])
                epoch_drop_reasons.extend(train_metrics["drop_reasons"])
                epoch_window_count += 1

                if adaptive_runtime is not None and window_assignment is not None:
                    if not window_assignment.is_last_window_for_epoch:
                        adaptive_runtime.schedule_prefetch(next_adaptive_window_id)
                        continue
                break

            train_metrics = _finalize_train_metrics(epoch_train_accumulator)

            should_eval = (epoch_idx + 1) % config.optimization.eval_every_epochs == 0
            val_metrics = (
                _evaluate(
                    model,
                    dataloaders.val,
                    criterion,
                    device,
                    num_classes=config.model.num_classes,
                    max_batches=overrides.max_eval_batches,
                )
                if should_eval
                else {"loss": None, "accuracy": None, "macro_f1": None}
            )
            final_test_metrics = _evaluate(
                model,
                dataloaders.test,
                criterion,
                device,
                num_classes=config.model.num_classes,
                max_batches=overrides.max_eval_batches,
            )

            mixed_peer_updates_total += epoch_mixed_peer_updates
            dropped_peer_updates_total += epoch_dropped_peer_updates
            nonfinite_peer_updates_total += epoch_nonfinite_peer_updates
            failed_pushes_total += epoch_failed_pushes
            push_count_total += epoch_push_count
            max_observed_staleness = max(max_observed_staleness, epoch_max_staleness)
            state_digest = digest_state(extract_model_state(model))
            epoch_row = {
                "epoch": epoch_idx + 1,
                "local_train_loss": train_metrics["loss"],
                "local_train_accuracy": train_metrics["accuracy"],
                "local_train_macro_f1": train_metrics["macro_f1"],
                "samples_processed": train_metrics["samples_processed"],
                "duration_s": time.perf_counter() - epoch_started_perf,
                "mixed_peer_updates": epoch_mixed_peer_updates,
                "mixed_senders": sorted(epoch_mixed_senders),
                "dropped_peer_updates": epoch_dropped_peer_updates,
                "dropped_peer_senders": sorted(epoch_dropped_peer_senders),
                "drop_reasons": epoch_drop_reasons,
                "nonfinite_peer_updates": epoch_nonfinite_peer_updates,
                "max_staleness": epoch_max_staleness,
                "pushes_sent": epoch_push_count,
                "failed_pushes_total": failed_pushes_total,
                "window_count": epoch_window_count,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_loss": final_test_metrics["loss"],
                "test_accuracy": final_test_metrics["accuracy"],
                "test_macro_f1": final_test_metrics["macro_f1"],
                "state_digest": state_digest,
                "last_step": local_step,
            }
            if epoch_window_rows:
                epoch_row["windows"] = epoch_window_rows
            epoch_row["samples_per_s"] = safe_rate(
                epoch_row["samples_processed"], epoch_row["duration_s"]
            )
            epoch_history.append(epoch_row)

            if (epoch_idx + 1) % config.logging.save_every_round == 0:
                torch.save(
                    {
                        "epoch": epoch_idx + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": epoch_row,
                        "self_node_id": resolved.self_node_id,
                    },
                    checkpoint_dir / f"async-epoch-{epoch_idx + 1:03d}.pt",
                )

        metrics_ext = "json" if config.logging.metrics_format == "json" else "csv"
        write_metrics(
            log_dir / f"async_epoch_metrics.{metrics_ext}",
            epoch_history,
            config.logging.metrics_format,
        )
        run_duration_s = time.perf_counter() - run_started_perf
        total_samples_processed = sum(row["samples_processed"] for row in epoch_history)
        best_val_accuracy = max(
            (
                row["val_accuracy"]
                for row in epoch_history
                if row.get("val_accuracy") is not None
            ),
            default=None,
        )
        best_val_macro_f1 = max(
            (
                row["val_macro_f1"]
                for row in epoch_history
                if row.get("val_macro_f1") is not None
            ),
            default=None,
        )
        data_plane_stats = adaptive_runtime.stats() if adaptive_runtime is not None else None
        scheduler_history_path = None
        if adaptive_runtime is not None:
            scheduler_history = adaptive_runtime.scheduler_history()
            scheduler_history_path = log_dir / "scheduler_history.csv"
            write_metrics(scheduler_history_path, scheduler_history, "csv")
        final_state_digest = epoch_history[-1]["state_digest"] if epoch_history else None
        completion_status = _coordinate_run_completion(
            resolved=resolved,
            server=server,
            completion=RunCompletionRecord(
                node_id=resolved.self_node_id,
                last_window_id=max(len(epoch_history) - 1, 0),
                total_samples_processed=total_samples_processed,
                final_state_digest=final_state_digest,
                completed_at=utc_now_iso(),
            ),
            timeout_s=_completion_timeout_s(resolved, overrides),
        )
        if overrides.shutdown_grace_s > 0:
            # Keep server alive so late-arriving followers can report completion.
            time.sleep(overrides.shutdown_grace_s)
        summary = {
            "run_id": run_id,
            "cluster_name": resolved.cluster.cluster_name,
            "self_node_id": resolved.self_node_id,
            "device": str(device),
            "dataset": config.dataset.name,
            "model": config.model.name,
            "mode": config.mode,
            "algorithm": config.algorithm,
            "history_kind": "epochs",
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "run_duration_s": run_duration_s,
            "cluster_node_count": len(resolved.cluster.nodes),
            "epoch_count": len(epoch_history),
            "local_train_sample_count": epoch_history[-1]["samples_processed"] if epoch_history else 0,
            "total_samples_processed": total_samples_processed,
            "effective_samples_per_s": safe_rate(total_samples_processed, run_duration_s),
            "best_val_accuracy": best_val_accuracy,
            "best_test_accuracy": max((row["test_accuracy"] for row in epoch_history), default=None),
            "best_val_macro_f1": best_val_macro_f1,
            "best_test_macro_f1": max((row["test_macro_f1"] for row in epoch_history), default=None),
            "metrics_file": f"async_epoch_metrics.{metrics_ext}",
            "scheduler_history_file": (
                str(scheduler_history_path) if scheduler_history_path is not None else None
            ),
            "transport_address": server.address,
            "advertise_address": _node_target(resolved.self_node.host, resolved.self_node.port),
            "neighbor_ids": [neighbor.id for neighbor in neighbors],
            "mixed_peer_updates_total": mixed_peer_updates_total,
            "dropped_peer_updates_total": dropped_peer_updates_total,
            "nonfinite_peer_updates_total": nonfinite_peer_updates_total,
            "max_observed_staleness": max_observed_staleness,
            "failed_pushes_total": failed_pushes_total,
            "push_count_total": push_count_total,
            "max_local_step": local_step,
            "epochs": epoch_history,
            "received_payload_count": server.snapshot().to_dict()["received_payload_count"],
            "final_test_metrics": final_test_metrics,
            "final_state_digest": final_state_digest,
            "data_plane_stats": data_plane_stats,
            "control_plane_role": (
                "bootstrap"
                if (resolved.cluster.bootstrap_node_id or resolved.self_node_id)
                == resolved.self_node_id
                else "follower"
            ),
            "completion_reported": completion_status["reported"],
            "completion_cluster_complete": completion_status["cluster_complete"],
            "completion_seen_count": completion_status["seen_count"],
            "missing_completion_nodes": completion_status["missing_node_ids"],
        }
        write_summary(log_dir / "async_run_summary.json", summary)
        return summary
    finally:
        if adaptive_runtime is not None:
            adaptive_runtime.close()
        server.stop(grace=0.0)
        if config_http_server is not None:
            config_http_server.stop()


def run_async_smoke(
    peer_count: int = 3,
    rounds: int = 2,
    *,
    storage_mode: str = "replicated",
    scheduler_mode: str = "static",
    rebalance_window_batches: int | None = None,
    fake_train_size: int | None = None,
) -> dict:
    if peer_count < 2:
        raise ValueError("peer_count must be at least 2")

    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()
    processes = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cluster_path, training_paths, node_ids = _write_async_smoke_configs(
            tmp_path=tmp_path,
            peer_count=peer_count,
            storage_mode=storage_mode,
            scheduler_mode=scheduler_mode,
            rebalance_window_batches=rebalance_window_batches,
            fake_train_size=fake_train_size,
        )

        try:
            for node_id in node_ids:
                round_delay_s = 0.8 if node_id == node_ids[-1] else 0.0
                process = ctx.Process(
                    target=_async_worker_process,
                    args=(
                        cluster_path,
                        training_paths[node_id],
                        node_id,
                        results_queue,
                        AsyncRunOverrides(
                            rounds=rounds,
                            max_local_batches=2,
                            max_eval_batches=1,
                            run_name="async-smoke",
                            transport_timeout_s=20.0,
                            bind_host="127.0.0.1",
                            round_delay_s=round_delay_s,
                            shutdown_grace_s=2.5,
                        ),
                    ),
                )
                process.start()
                processes.append(process)

            results = {}
            for _ in node_ids:
                item = results_queue.get(timeout=90.0)
                results[item["self_node_id"]] = item

            exit_codes = []
            for process in processes:
                process.join(timeout=30.0)
                exit_codes.append(process.exitcode)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)

            return {
                "peer_count": peer_count,
                "configured_epochs": rounds,
                "node_results": results,
                "exit_codes": exit_codes,
            }
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)


def _async_worker_process(
    cluster_path: Path,
    training_path: Path,
    self_node_id: str,
    results_queue: mp.Queue,
    overrides: AsyncRunOverrides,
) -> None:
    try:
        resolved = load_resolved_config(cluster_path, training_path, self_node_id)
        result = run_async_worker(resolved, overrides=overrides)
    except Exception:
        result = {
            "self_node_id": self_node_id,
            "error": traceback.format_exc(),
        }
    results_queue.put(result)


def _mix_with_latest_peer_payloads(
    *,
    current_state: dict[str, torch.Tensor],
    current_version: int,
    local_sample_count: int,
    server: PeerServer,
    neighbor_ids: list[str],
    base_alpha: float,
    max_staleness: int,
    push_interval_steps: int,
    last_mixed_payload_ids: dict[str, str] | None = None,
) -> dict:
    require_state_finite(current_state, context=f"current async state at version={current_version}")
    merged_state = {name: tensor.clone() for name, tensor in current_state.items()}
    snapshot = server.snapshot()
    eligible = []
    dropped_peer_updates = 0
    nonfinite_peer_updates = 0
    dropped_peer_senders: set[str] = set()
    drop_reasons: list[str] = []

    def drop_payload(sender_id: str, payload_id: str, reason: str) -> None:
        nonlocal dropped_peer_updates
        dropped_peer_updates += 1
        dropped_peer_senders.add(sender_id)
        drop_reasons.append(f"{sender_id}:{reason}")
        if last_mixed_payload_ids is not None:
            last_mixed_payload_ids[sender_id] = payload_id

    for summary in snapshot.payloads:
        if summary.sender_node_id not in neighbor_ids:
            continue
        if summary.payload_kind != "async_weights":
            continue
        if (
            last_mixed_payload_ids is not None
            and last_mixed_payload_ids.get(summary.sender_node_id) == summary.payload_id
        ):
            continue
        version_gap = abs(current_version - summary.model_version)
        if version_gap > max_staleness:
            drop_payload(
                summary.sender_node_id,
                summary.payload_id,
                f"version_gap:{version_gap}>max:{max_staleness}",
            )
            continue
        payload = server.get_payload(summary.sender_node_id, summary.payload_id)
        if payload is None:
            continue
        peer_keys = set(payload.tensors)
        local_keys = set(current_state)
        if peer_keys != local_keys:
            missing = sorted(local_keys - peer_keys)
            extra = sorted(peer_keys - local_keys)
            reason = (
                f"state_keys_mismatch:missing={missing[:4]}:extra={extra[:4]}"
            )
            drop_payload(summary.sender_node_id, summary.payload_id, reason)
            continue
        incompatible_tensor = next(
            (
                name
                for name in current_state
                if current_state[name].shape != payload.tensors[name].shape
                or current_state[name].dtype != payload.tensors[name].dtype
            ),
            None,
        )
        if incompatible_tensor is not None:
            local_tensor = current_state[incompatible_tensor]
            peer_tensor = payload.tensors[incompatible_tensor]
            drop_payload(
                summary.sender_node_id,
                summary.payload_id,
                "state_tensor_mismatch:"
                f"{incompatible_tensor}:"
                f"local_shape={tuple(local_tensor.shape)}:"
                f"peer_shape={tuple(peer_tensor.shape)}:"
                f"local_dtype={local_tensor.dtype}:"
                f"peer_dtype={peer_tensor.dtype}",
            )
            continue
        peer_report = check_state_finite(payload.tensors)
        if not peer_report.ok:
            nonfinite_peer_updates += 1
            drop_payload(
                summary.sender_node_id,
                summary.payload_id,
                f"nonfinite_payload:{peer_report.format_summary()}",
            )
            continue
        eligible.append((summary.sender_node_id, summary.payload_id, version_gap, payload))

    eligible.sort(key=lambda item: item[0])
    mixed_senders = []
    observed_version_gap = 0
    staleness_scale = float(max(push_interval_steps, 1))
    for sender_id, payload_id, version_gap, payload in eligible:
        peer_sample_count = max(payload.metadata.sample_count, 1)
        local_weight = max(local_sample_count, 1)
        peer_ratio = peer_sample_count / float(local_weight + peer_sample_count)
        alpha = min(
            1.0,
            max(0.0, base_alpha * peer_ratio / (1.0 + version_gap / staleness_scale)),
        )
        proposed_state = {name: tensor.clone() for name, tensor in merged_state.items()}
        for name in proposed_state:
            if torch.is_floating_point(proposed_state[name]):
                proposed_state[name] = (
                    proposed_state[name] * (1.0 - alpha) + payload.tensors[name] * alpha
                )
        merged_report = check_state_finite(proposed_state)
        if not merged_report.ok:
            nonfinite_peer_updates += 1
            drop_payload(
                sender_id,
                payload_id,
                f"nonfinite_merged_state:{merged_report.format_summary()}",
            )
            continue
        merged_state = proposed_state
        mixed_senders.append(sender_id)
        observed_version_gap = max(observed_version_gap, version_gap)
        if last_mixed_payload_ids is not None:
            last_mixed_payload_ids[sender_id] = payload_id

    return {
        "state": merged_state,
        "mixed_peer_updates": len(mixed_senders),
        "mixed_senders": mixed_senders,
        "max_staleness": observed_version_gap,
        "dropped_peer_updates": dropped_peer_updates,
        "dropped_peer_senders": sorted(dropped_peer_senders),
        "drop_reasons": drop_reasons,
        "nonfinite_peer_updates": nonfinite_peer_updates,
    }


def _push_payload_best_effort(
    *,
    target: str,
    sender_node_id: str,
    payload: PeerPayload,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        client = PeerClient(target)
        try:
            remaining_s = deadline - time.time()
            if remaining_s <= 0:
                break
            client.push_payload(
                payload,
                timeout_s=min(_MAX_PUSH_ATTEMPT_TIMEOUT_S, max(0.5, remaining_s)),
            )
            return True
        except Exception:
            time.sleep(0.1)
        finally:
            client.close()
    return False


def _write_async_smoke_configs(
    tmp_path: Path,
    peer_count: int,
    *,
    storage_mode: str = "replicated",
    scheduler_mode: str = "static",
    rebalance_window_batches: int | None = None,
    fake_train_size: int | None = None,
) -> tuple[Path, dict[str, Path], list[str]]:
    node_ids = [f"node-{idx + 1}" for idx in range(peer_count)]
    ports = [_find_free_port() for _ in range(peer_count)]
    cluster_payload = {
        "cluster_name": "async-smoke",
        "transport": "grpc",
        "overlay_network": "none",
        "tls_enabled": False,
        "bootstrap_node_id": node_ids[0],
        "nodes": [],
    }
    for idx, node_id in enumerate(node_ids):
        neighbors = [neighbor_id for neighbor_id in node_ids if neighbor_id != node_id]
        cluster_payload["nodes"].append(
            {
                "id": node_id,
                "host": "127.0.0.1",
                "port": ports[idx],
                "platform": "linux",
                "neighbors": neighbors,
                "weight": 1.0,
                "resources": {
                    "cpu_cores": 2,
                    "accelerator": "cpu",
                    "relative_speed": 1.0,
                },
            }
        )

    training_payload = load_yaml(PROJECT_ROOT / "configs" / "training.local-smoke.yaml")
    training_payload["device_preference"] = ["cpu"]
    training_payload["mode"] = "async"
    training_payload["async"]["enabled"] = True
    training_payload["async"]["push_interval_steps"] = 1
    training_payload["async"]["max_staleness"] = 2
    training_payload["async"]["mixing_alpha"] = 0.7
    training_payload["sync"]["enabled"] = False
    training_payload["dataset"]["partitioning"] = "homogeneous"
    training_payload["dataset"]["scheduler_mode"] = scheduler_mode
    training_payload["dataset"]["rebalance_window_batches"] = (
        rebalance_window_batches if rebalance_window_batches is not None else 2
    )
    training_payload["dataset"]["throughput_ema"] = 0.0
    training_payload["dataset"]["warmup_windows"] = 1
    training_payload["dataset"]["min_local_shards"] = 1
    if fake_train_size is not None:
        training_payload["dataset"]["fake_train_size"] = fake_train_size
    training_payload["logging"]["log_dir"] = str(tmp_path / "logs")
    training_payload["logging"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    training_payload["optimization"]["eval_every_epochs"] = 1

    cluster_path = tmp_path / "cluster.async-smoke.yaml"
    with cluster_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cluster_payload, handle, sort_keys=False)

    training_paths: dict[str, Path] = {}
    if storage_mode == "replicated":
        training_path = tmp_path / "training.async-smoke.yaml"
        with training_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(training_payload, handle, sort_keys=False)
        for node_id in node_ids:
            training_paths[node_id] = training_path
        return cluster_path, training_paths, node_ids

    if storage_mode != "micro_shards":
        raise ValueError(f"Unsupported storage_mode for async smoke: {storage_mode}")

    training_payload["dataset"]["storage_mode"] = "micro_shards"
    training_payload["dataset"]["shard_samples"] = 4
    bootstrap_manifest_path = tmp_path / node_ids[0] / "manifest.json"
    training_payload["dataset"]["manifest_path"] = str(bootstrap_manifest_path)
    training_payload["dataset"]["cache_dir"] = str(bootstrap_manifest_path.parent)
    bootstrap_training_path = tmp_path / node_ids[0] / "training.async-smoke.micro.yaml"
    bootstrap_training_path.parent.mkdir(parents=True, exist_ok=True)
    with bootstrap_training_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(training_payload, handle, sort_keys=False)
    build_dataset_shards(load_resolved_config(cluster_path, bootstrap_training_path, node_ids[0]).training)
    manifest = load_manifest(bootstrap_manifest_path)

    for node_id in node_ids:
        node_dir = tmp_path / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        node_training_payload = load_yaml(bootstrap_training_path)
        node_training_payload["dataset"]["manifest_path"] = str(node_dir / "manifest.json")
        node_training_payload["dataset"]["cache_dir"] = str(
            node_dir if node_id == node_ids[0] else node_dir / "cache"
        )
        node_training_payload["logging"]["log_dir"] = str(tmp_path / "logs")
        node_training_payload["logging"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
        node_training_path = node_dir / "training.async-smoke.micro.yaml"
        with node_training_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(node_training_payload, handle, sort_keys=False)
        if node_id == node_ids[0]:
            save_manifest(manifest, node_dir / "manifest.json")
        training_paths[node_id] = node_training_path

    return cluster_path, training_paths, node_ids


def _wait_for_neighbors(self_node_id: str, neighbors: list, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    pending = {neighbor.id: _node_target(neighbor.host, neighbor.port) for neighbor in neighbors}
    while pending:
        if time.time() > deadline:
            missing = ", ".join(sorted(pending))
            raise TimeoutError(f"Node '{self_node_id}' timed out waiting for neighbors: {missing}")

        for neighbor_id in list(pending):
            client = PeerClient(pending[neighbor_id])
            try:
                client.ping(sender_node_id=self_node_id, timeout_s=1.0)
            except Exception:
                time.sleep(0.1)
            else:
                pending.pop(neighbor_id)
            finally:
                client.close()


def _tensor_finite_status(tensor: torch.Tensor) -> str:
    tensor_cpu = tensor.detach().to("cpu")
    finite_mask = torch.isfinite(tensor_cpu)
    finite_count = int(finite_mask.sum().item())
    total_count = tensor_cpu.numel()
    if finite_count == 0:
        return f"finite={finite_count}/{total_count}, min=None, max=None"
    finite_values = tensor_cpu[finite_mask]
    return (
        f"finite={finite_count}/{total_count}, "
        f"min={float(finite_values.min().item())}, "
        f"max={float(finite_values.max().item())}"
    )


def _train_async_window(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    max_batches: int | None,
    start_step: int,
    run_id: str,
    self_node_id: str,
    server: PeerServer,
    neighbors: list,
    push_fanout: int,
    push_interval_steps: int,
    base_alpha: float,
    max_staleness: int,
    transport_timeout_s: float,
    last_mixed_payload_ids: dict[str, str],
) -> dict[str, float | int | list[str]]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    processed_batches = 0
    confusion = init_confusion_matrix(num_classes)
    current_step = start_step
    samples_since_push = 0
    pushes_sent = 0
    failed_pushes = 0
    mixed_peer_updates = 0
    max_observed_staleness = 0
    mixed_senders_seen: set[str] = set()
    dropped_peer_updates = 0
    nonfinite_peer_updates = 0
    dropped_peer_senders_seen: set[str] = set()
    drop_reasons: list[str] = []

    for inputs, targets in loader:
        targets_cpu = targets.detach().to("cpu", dtype=torch.int64).reshape(-1)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError(
                "Non-finite logits during async training: "
                f"node_id={self_node_id}, step={current_step + 1}, "
                f"batch_index={processed_batches}, {_tensor_finite_status(logits)}"
            )
        loss = criterion(logits, targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                "Non-finite loss during async training: "
                f"node_id={self_node_id}, step={current_step + 1}, "
                f"batch_index={processed_batches}, loss={float(loss.detach().to('cpu').item())}, "
                f"logits={_tensor_finite_status(logits)}"
            )
        loss.backward()
        optimizer.step()

        batch_size = targets_cpu.numel()
        total_loss += loss.item() * batch_size
        preds_cpu = logits.argmax(dim=1).detach().to("cpu", dtype=torch.int64).reshape(-1)
        correct += (preds_cpu == targets_cpu).sum().item()
        total += batch_size
        update_confusion_matrix(confusion, logits, targets_cpu)
        processed_batches += 1
        current_step += 1
        samples_since_push += batch_size

        if current_step % push_interval_steps == 0:
            exchange_result = _exchange_async_update(
                model=model,
                device=device,
                optimizer=optimizer,
                current_step=current_step,
                sample_count=samples_since_push,
                run_id=run_id,
                self_node_id=self_node_id,
                server=server,
                neighbors=neighbors,
                push_fanout=push_fanout,
                push_interval_steps=push_interval_steps,
                base_alpha=base_alpha,
                max_staleness=max_staleness,
                transport_timeout_s=transport_timeout_s,
                last_mixed_payload_ids=last_mixed_payload_ids,
            )
            pushes_sent += exchange_result["pushes_sent"]
            failed_pushes += exchange_result["failed_pushes"]
            mixed_peer_updates += exchange_result["mixed_peer_updates"]
            max_observed_staleness = max(
                max_observed_staleness, exchange_result["max_staleness"]
            )
            mixed_senders_seen.update(exchange_result["mixed_senders"])
            dropped_peer_updates += exchange_result["dropped_peer_updates"]
            nonfinite_peer_updates += exchange_result["nonfinite_peer_updates"]
            dropped_peer_senders_seen.update(exchange_result["dropped_peer_senders"])
            drop_reasons.extend(exchange_result["drop_reasons"])
            samples_since_push = 0

        if max_batches is not None and processed_batches >= max_batches:
            break

    if processed_batches > 0 and (samples_since_push > 0 or pushes_sent == 0):
        exchange_result = _exchange_async_update(
            model=model,
            device=device,
            optimizer=optimizer,
            current_step=current_step,
            sample_count=max(samples_since_push, 1),
            run_id=run_id,
            self_node_id=self_node_id,
            server=server,
            neighbors=neighbors,
            push_fanout=push_fanout,
            push_interval_steps=push_interval_steps,
            base_alpha=base_alpha,
            max_staleness=max_staleness,
            transport_timeout_s=transport_timeout_s,
            last_mixed_payload_ids=last_mixed_payload_ids,
        )
        pushes_sent += exchange_result["pushes_sent"]
        failed_pushes += exchange_result["failed_pushes"]
        mixed_peer_updates += exchange_result["mixed_peer_updates"]
        max_observed_staleness = max(max_observed_staleness, exchange_result["max_staleness"])
        mixed_senders_seen.update(exchange_result["mixed_senders"])
        dropped_peer_updates += exchange_result["dropped_peer_updates"]
        nonfinite_peer_updates += exchange_result["nonfinite_peer_updates"]
        dropped_peer_senders_seen.update(exchange_result["dropped_peer_senders"])
        drop_reasons.extend(exchange_result["drop_reasons"])

    summary_metrics = summarize_classification_metrics(
        total_loss=total_loss,
        correct=correct,
        total=total,
        confusion=confusion,
    )
    return {
        **summary_metrics,
        "total_loss_raw": total_loss,
        "correct_count": correct,
        "confusion": confusion,
        "mixed_peer_updates": mixed_peer_updates,
        "mixed_senders": sorted(mixed_senders_seen),
        "max_staleness": max_observed_staleness,
        "dropped_peer_updates": dropped_peer_updates,
        "dropped_peer_senders": sorted(dropped_peer_senders_seen),
        "drop_reasons": drop_reasons,
        "nonfinite_peer_updates": nonfinite_peer_updates,
        "pushes_sent": pushes_sent,
        "failed_pushes": failed_pushes,
        "last_step": current_step,
    }


def _init_train_accumulator(num_classes: int) -> dict[str, object]:
    return {
        "total_loss_raw": 0.0,
        "correct_count": 0,
        "samples_processed": 0,
        "confusion": init_confusion_matrix(num_classes),
    }


def _accumulate_train_metrics(accumulator: dict[str, object], window_metrics: dict[str, object]) -> None:
    accumulator["total_loss_raw"] = float(accumulator["total_loss_raw"]) + float(
        window_metrics["total_loss_raw"]
    )
    accumulator["correct_count"] = int(accumulator["correct_count"]) + int(
        window_metrics["correct_count"]
    )
    accumulator["samples_processed"] = int(accumulator["samples_processed"]) + int(
        window_metrics["samples_processed"]
    )
    accumulator["confusion"] += window_metrics["confusion"]


def _finalize_train_metrics(accumulator: dict[str, object]) -> dict[str, float]:
    total_loss = float(accumulator["total_loss_raw"])
    correct = int(accumulator["correct_count"])
    total = int(accumulator["samples_processed"])
    confusion = accumulator["confusion"]
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1_from_confusion(confusion),
        "samples_processed": total,
    }


def _exchange_async_update(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    current_step: int,
    sample_count: int,
    run_id: str,
    self_node_id: str,
    server: PeerServer,
    neighbors: list,
    push_fanout: int,
    push_interval_steps: int,
    base_alpha: float,
    max_staleness: int,
    transport_timeout_s: float,
    last_mixed_payload_ids: dict[str, str],
) -> dict:
    current_state = extract_model_state(model)
    require_state_finite(
        current_state,
        context=f"node_id={self_node_id}, step={current_step}, outgoing async state",
    )
    payload = PeerPayload(
        metadata=PayloadMetadata(
            sender_node_id=self_node_id,
            payload_id=f"{run_id}:step:{current_step:05d}",
            payload_kind="async_weights",
            model_version=current_step,
            step=current_step,
            sample_count=max(sample_count, 1),
        ),
        tensors=current_state,
    )

    pushes_sent = 0
    failed_pushes = 0
    selected_neighbors = _select_push_neighbors(
        neighbors,
        push_fanout=push_fanout,
        current_step=current_step,
        self_node_id=self_node_id,
    )
    for neighbor in selected_neighbors:
        pushed = _push_payload_best_effort(
            target=_node_target(neighbor.host, neighbor.port),
            sender_node_id=self_node_id,
            payload=payload,
            timeout_s=transport_timeout_s,
        )
        if pushed:
            pushes_sent += 1
        else:
            failed_pushes += 1

    merge_result = _mix_with_latest_peer_payloads(
        current_state=current_state,
        current_version=current_step,
        local_sample_count=max(sample_count, 1),
        server=server,
        neighbor_ids=[neighbor.id for neighbor in neighbors],
        base_alpha=base_alpha,
        max_staleness=max_staleness,
        push_interval_steps=push_interval_steps,
        last_mixed_payload_ids=last_mixed_payload_ids,
    )
    if merge_result["mixed_peer_updates"] > 0:
        require_state_finite(
            merge_result["state"],
            context=f"node_id={self_node_id}, step={current_step}, merged async state",
        )
        load_model_state(model, merge_result["state"], device)
        optimizer.state.clear()
    merge_result["pushes_sent"] = pushes_sent
    merge_result["failed_pushes"] = failed_pushes
    return merge_result


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    processed_batches = 0
    confusion = init_confusion_matrix(num_classes)

    for inputs, targets in loader:
        targets_cpu = targets.detach().to("cpu", dtype=torch.int64).reshape(-1)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets_cpu.numel()
        total_loss += loss.item() * batch_size
        preds_cpu = logits.argmax(dim=1).detach().to("cpu", dtype=torch.int64).reshape(-1)
        correct += (preds_cpu == targets_cpu).sum().item()
        total += batch_size
        update_confusion_matrix(confusion, logits, targets_cpu)
        processed_batches += 1

        if max_batches is not None and processed_batches >= max_batches:
            break

    metrics = summarize_classification_metrics(
        total_loss=total_loss,
        correct=correct,
        total=total,
        confusion=confusion,
    )
    metrics.pop("samples_processed", None)
    return metrics


def _node_target(host: str, port: int) -> str:
    return f"{host}:{port}"


def _select_push_neighbors(
    neighbors: list,
    *,
    push_fanout: int,
    current_step: int,
    self_node_id: str,
) -> list:
    ordered_neighbors = sorted(neighbors, key=lambda item: item.id)
    if push_fanout <= 0 or push_fanout >= len(ordered_neighbors):
        return ordered_neighbors

    offset_seed = current_step + sum(ord(char) for char in self_node_id)
    offset = offset_seed % len(ordered_neighbors)
    rotated = ordered_neighbors[offset:] + ordered_neighbors[:offset]
    return rotated[:push_fanout]


def _completion_timeout_s(resolved: ResolvedConfig, overrides: AsyncRunOverrides) -> float:
    cluster_wait_floor = max(10.0, overrides.transport_timeout_s * len(resolved.cluster.nodes))
    return max(cluster_wait_floor, overrides.shutdown_grace_s)


def _coordinate_run_completion(
    *,
    resolved: ResolvedConfig,
    server: PeerServer,
    completion: RunCompletionRecord,
    timeout_s: float,
) -> dict[str, object]:
    expected_node_ids = [node.id for node in resolved.cluster.nodes]
    bootstrap_node_id = resolved.cluster.bootstrap_node_id or resolved.self_node_id
    is_bootstrap = resolved.self_node_id == bootstrap_node_id

    if is_bootstrap:
        server.store_run_completion(completion)
        cluster_complete = server.wait_for_run_completions(
            node_ids=expected_node_ids,
            timeout_s=timeout_s,
        )
        completions = server.get_run_completions()
        seen_node_ids = {item.node_id for item in completions}
        return {
            "reported": True,
            "cluster_complete": cluster_complete,
            "seen_count": len(seen_node_ids),
            "missing_node_ids": sorted(
                node_id for node_id in expected_node_ids if node_id not in seen_node_ids
            ),
        }

    reported = _report_run_completion_with_retry(
        target=_node_target(
            resolved.cluster.get_node(bootstrap_node_id).host,
            resolved.cluster.get_node(bootstrap_node_id).port,
        ),
        completion=completion,
        timeout_s=timeout_s,
    )
    return {
        "reported": reported,
        "cluster_complete": None,
        "seen_count": 1 if reported else 0,
        "missing_node_ids": [] if reported else [resolved.self_node_id],
    }


def _report_run_completion_with_retry(
    *,
    target: str,
    completion: RunCompletionRecord,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        client = PeerClient(target)
        try:
            remaining_s = deadline - time.time()
            if remaining_s <= 0:
                break
            client.report_run_completion(
                completion,
                timeout_s=min(_MAX_PUSH_ATTEMPT_TIMEOUT_S, max(0.5, remaining_s)),
            )
            return True
        except Exception:
            time.sleep(0.1)
        finally:
            client.close()
    return False


def _resolve_async_epoch_count(config, overrides: AsyncRunOverrides) -> int:
    if overrides.epochs is not None:
        return overrides.epochs
    if overrides.rounds is not None:
        return overrides.rounds
    return config.optimization.epochs


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
