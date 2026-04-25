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
    summarize_classification_metrics,
    update_confusion_matrix,
)
from decentr_my_own.training.seed import set_global_seed
from decentr_my_own.training.state_ops import (
    digest_state,
    extract_model_state,
    load_model_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AsyncRunOverrides:
    rounds: int = 2
    max_local_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str | None = None
    transport_timeout_s: float = 15.0
    bind_host: str | None = None
    round_delay_s: float = 0.0
    shutdown_grace_s: float = 2.0


def run_async_worker(
    resolved: ResolvedConfig, overrides: AsyncRunOverrides | None = None
) -> dict:
    overrides = overrides or AsyncRunOverrides()
    config = resolved.training
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

        round_history = []
        mixed_peer_updates_total = 0
        max_observed_staleness = 0
        failed_pushes_total = 0
        push_count_total = 0
        local_step = 0
        last_mixed_payload_ids: dict[str, str] = {}
        final_test_metrics = {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

        for round_idx in range(overrides.rounds):
            round_started_perf = time.perf_counter()
            if adaptive_runtime is not None:
                train_shard_ids = adaptive_runtime.get_window_shard_ids(round_idx)
            else:
                train_shard_ids = static_shard_ids
            dataloaders = build_local_dataloaders(
                resolved,
                pin_memory=pin_memory,
                train_shard_ids=train_shard_ids,
            )
            if dataloaders.train_sampler is not None:
                dataloaders.train_sampler.set_epoch(round_idx)
            if adaptive_runtime is not None and round_idx + 1 < overrides.rounds:
                adaptive_runtime.schedule_prefetch(round_idx + 1)

            train_started_perf = time.perf_counter()
            train_metrics = _train_async_window(
                model,
                dataloaders.train,
                optimizer,
                criterion,
                device,
                num_classes=config.model.num_classes,
                max_batches=overrides.max_local_batches or config.optimization.local_steps,
                start_step=local_step,
                run_id=run_id,
                self_node_id=resolved.self_node_id,
                server=server,
                neighbors=neighbors,
                push_interval_steps=config.async_config.push_interval_steps,
                base_alpha=config.async_config.mixing_alpha,
                max_staleness=config.async_config.max_staleness,
                transport_timeout_s=min(1.0, overrides.transport_timeout_s),
                last_mixed_payload_ids=last_mixed_payload_ids,
            )
            local_step = train_metrics["last_step"]
            if overrides.round_delay_s > 0:
                time.sleep(overrides.round_delay_s)
            train_duration_s = time.perf_counter() - train_started_perf
            if adaptive_runtime is not None:
                adaptive_runtime.report_window(
                    window_id=round_idx,
                    samples_processed=train_metrics["samples_processed"],
                    duration_s=train_duration_s,
                )

            should_eval = (round_idx + 1) % config.optimization.eval_every_epochs == 0
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

            mixed_peer_updates_total += train_metrics["mixed_peer_updates"]
            failed_pushes_total += train_metrics["failed_pushes"]
            push_count_total += train_metrics["pushes_sent"]
            max_observed_staleness = max(max_observed_staleness, train_metrics["max_staleness"])
            state_digest = digest_state(extract_model_state(model))
            round_row = {
                "round": round_idx + 1,
                "local_train_loss": train_metrics["loss"],
                "local_train_accuracy": train_metrics["accuracy"],
                "local_train_macro_f1": train_metrics["macro_f1"],
                "samples_processed": train_metrics["samples_processed"],
                "duration_s": time.perf_counter() - round_started_perf,
                "mixed_peer_updates": train_metrics["mixed_peer_updates"],
                "mixed_senders": train_metrics["mixed_senders"],
                "max_staleness": train_metrics["max_staleness"],
                "pushes_sent": train_metrics["pushes_sent"],
                "failed_pushes_total": failed_pushes_total,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_loss": final_test_metrics["loss"],
                "test_accuracy": final_test_metrics["accuracy"],
                "test_macro_f1": final_test_metrics["macro_f1"],
                "state_digest": state_digest,
                "last_step": train_metrics["last_step"],
            }
            round_row["samples_per_s"] = safe_rate(
                round_row["samples_processed"], round_row["duration_s"]
            )
            round_history.append(round_row)

            if (round_idx + 1) % config.logging.save_every_round == 0:
                torch.save(
                    {
                        "round": round_idx + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": round_row,
                        "self_node_id": resolved.self_node_id,
                    },
                    checkpoint_dir / f"async-round-{round_idx + 1:03d}.pt",
                )

        if overrides.shutdown_grace_s > 0:
            time.sleep(overrides.shutdown_grace_s)

        metrics_ext = "json" if config.logging.metrics_format == "json" else "csv"
        write_metrics(
            log_dir / f"async_round_metrics.{metrics_ext}",
            round_history,
            config.logging.metrics_format,
        )
        run_duration_s = time.perf_counter() - run_started_perf
        total_samples_processed = sum(row["samples_processed"] for row in round_history)
        best_val_accuracy = max(
            (
                row["val_accuracy"]
                for row in round_history
                if row.get("val_accuracy") is not None
            ),
            default=None,
        )
        best_val_macro_f1 = max(
            (
                row["val_macro_f1"]
                for row in round_history
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
        summary = {
            "run_id": run_id,
            "cluster_name": resolved.cluster.cluster_name,
            "self_node_id": resolved.self_node_id,
            "device": str(device),
            "dataset": config.dataset.name,
            "model": config.model.name,
            "mode": config.mode,
            "algorithm": config.algorithm,
            "history_kind": "rounds",
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "run_duration_s": run_duration_s,
            "round_count": len(round_history),
            "train_sample_count": dataloaders.train_sample_count,
            "total_samples_processed": total_samples_processed,
            "effective_samples_per_s": safe_rate(total_samples_processed, run_duration_s),
            "best_val_accuracy": best_val_accuracy,
            "best_test_accuracy": max((row["test_accuracy"] for row in round_history), default=None),
            "best_val_macro_f1": best_val_macro_f1,
            "best_test_macro_f1": max((row["test_macro_f1"] for row in round_history), default=None),
            "metrics_file": f"async_round_metrics.{metrics_ext}",
            "scheduler_history_file": (
                str(scheduler_history_path) if scheduler_history_path is not None else None
            ),
            "transport_address": server.address,
            "advertise_address": _node_target(resolved.self_node.host, resolved.self_node.port),
            "neighbor_ids": [neighbor.id for neighbor in neighbors],
            "mixed_peer_updates_total": mixed_peer_updates_total,
            "max_observed_staleness": max_observed_staleness,
            "failed_pushes_total": failed_pushes_total,
            "push_count_total": push_count_total,
            "max_local_step": local_step,
            "rounds": round_history,
            "received_payload_count": server.snapshot().to_dict()["received_payload_count"],
            "final_test_metrics": final_test_metrics,
            "final_state_digest": round_history[-1]["state_digest"] if round_history else None,
            "data_plane_stats": data_plane_stats,
        }
        write_summary(log_dir / "async_run_summary.json", summary)
        return summary
    finally:
        if adaptive_runtime is not None:
            adaptive_runtime.close()
        server.stop(grace=0.0)


def run_async_smoke(
    peer_count: int = 3,
    rounds: int = 2,
    *,
    storage_mode: str = "replicated",
    scheduler_mode: str = "static",
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
                "rounds": rounds,
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
    last_mixed_payload_ids: dict[str, str] | None = None,
) -> dict:
    merged_state = {name: tensor.clone() for name, tensor in current_state.items()}
    snapshot = server.snapshot()
    eligible = []
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
        staleness = max(0, current_version - summary.model_version)
        if staleness > max_staleness:
            continue
        payload = server.get_payload(summary.sender_node_id)
        if payload is None:
            continue
        eligible.append((summary.sender_node_id, summary.payload_id, staleness, payload))

    eligible.sort(key=lambda item: item[0])
    mixed_senders = []
    observed_staleness = 0
    for sender_id, payload_id, staleness, payload in eligible:
        peer_sample_count = max(payload.metadata.sample_count, 1)
        local_weight = max(local_sample_count, 1)
        peer_ratio = peer_sample_count / float(local_weight + peer_sample_count)
        alpha = min(1.0, base_alpha * peer_ratio / float(1 + staleness))
        for name in merged_state:
            merged_state[name] = merged_state[name] * (1.0 - alpha) + payload.tensors[name] * alpha
        mixed_senders.append(sender_id)
        observed_staleness = max(observed_staleness, staleness)
        if last_mixed_payload_ids is not None:
            last_mixed_payload_ids[sender_id] = payload_id

    return {
        "state": merged_state,
        "mixed_peer_updates": len(mixed_senders),
        "mixed_senders": mixed_senders,
        "max_staleness": observed_staleness,
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
            client.push_payload(payload, timeout_s=min(2.0, max(0.5, deadline - time.time())))
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
    training_payload["dataset"]["rebalance_window_batches"] = 2
    training_payload["dataset"]["throughput_ema"] = 0.0
    training_payload["dataset"]["warmup_windows"] = 1
    training_payload["dataset"]["min_local_shards"] = 1
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


def _train_async_window(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    max_batches: int,
    start_step: int,
    run_id: str,
    self_node_id: str,
    server: PeerServer,
    neighbors: list,
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

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size
        update_confusion_matrix(confusion, logits, targets)
        processed_batches += 1
        current_step += 1
        samples_since_push += batch_size

        if current_step % push_interval_steps == 0:
            exchange_result = _exchange_async_update(
                model=model,
                device=device,
                current_step=current_step,
                sample_count=samples_since_push,
                run_id=run_id,
                self_node_id=self_node_id,
                server=server,
                neighbors=neighbors,
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
            samples_since_push = 0

        if processed_batches >= max_batches:
            break

    if processed_batches > 0 and (samples_since_push > 0 or pushes_sent == 0):
        exchange_result = _exchange_async_update(
            model=model,
            device=device,
            current_step=current_step,
            sample_count=max(samples_since_push, 1),
            run_id=run_id,
            self_node_id=self_node_id,
            server=server,
            neighbors=neighbors,
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

    return {
        **summarize_classification_metrics(
            total_loss=total_loss,
            correct=correct,
            total=total,
            confusion=confusion,
        ),
        "mixed_peer_updates": mixed_peer_updates,
        "mixed_senders": sorted(mixed_senders_seen),
        "max_staleness": max_observed_staleness,
        "pushes_sent": pushes_sent,
        "failed_pushes": failed_pushes,
        "last_step": current_step,
    }


def _exchange_async_update(
    model: nn.Module,
    device: torch.device,
    *,
    current_step: int,
    sample_count: int,
    run_id: str,
    self_node_id: str,
    server: PeerServer,
    neighbors: list,
    base_alpha: float,
    max_staleness: int,
    transport_timeout_s: float,
    last_mixed_payload_ids: dict[str, str],
) -> dict:
    current_state = extract_model_state(model)
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
    for neighbor in neighbors:
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
        last_mixed_payload_ids=last_mixed_payload_ids,
    )
    load_model_state(model, merge_result["state"], device)
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
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
        update_confusion_matrix(confusion, logits, targets)
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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
