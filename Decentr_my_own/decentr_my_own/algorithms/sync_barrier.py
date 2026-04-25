from __future__ import annotations

import multiprocessing as mp
import socket
import tempfile
import time
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
    add_state_delta,
    compute_model_delta,
    digest_state,
    extract_model_state,
    load_model_state,
    weighted_average_states,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SyncRunOverrides:
    rounds: int = 1
    max_local_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str | None = None
    transport_timeout_s: float = 15.0
    bind_host: str | None = None


def run_sync_worker(
    resolved: ResolvedConfig, overrides: SyncRunOverrides | None = None
) -> dict:
    overrides = overrides or SyncRunOverrides()
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

            base_state = extract_model_state(model)
            train_started_perf = time.perf_counter()
            train_metrics = _train_for_steps(
                model,
                dataloaders.train,
                optimizer,
                criterion,
                device,
                num_classes=config.model.num_classes,
                max_batches=overrides.max_local_batches or config.optimization.local_steps,
            )
            train_duration_s = time.perf_counter() - train_started_perf
            if adaptive_runtime is not None:
                adaptive_runtime.report_window(
                    window_id=round_idx,
                    samples_processed=train_metrics["samples_processed"],
                    duration_s=train_duration_s,
                )
            current_state = extract_model_state(model)

            payload_id = f"{run_id}:round:{round_idx:03d}"
            local_payload_state = _build_payload_state(
                averaging=config.sync.averaging,
                current_state=current_state,
                base_state=base_state,
            )
            local_payload = PeerPayload(
                metadata=PayloadMetadata(
                    sender_node_id=resolved.self_node_id,
                    payload_id=payload_id,
                    payload_kind=f"sync_{config.sync.averaging}",
                    model_version=round_idx + 1,
                    step=round_idx + 1,
                    sample_count=train_metrics["samples_processed"],
                ),
                tensors=local_payload_state,
            )

            for neighbor in neighbors:
                client = PeerClient(_node_target(neighbor.host, neighbor.port))
                try:
                    client.push_payload(local_payload, timeout_s=overrides.transport_timeout_s)
                finally:
                    client.close()

            received = server.wait_for_payload_ids(
                sender_node_ids=[neighbor.id for neighbor in neighbors],
                payload_id=payload_id,
                timeout_s=overrides.transport_timeout_s,
            )
            if not received:
                raise TimeoutError(
                    f"Node '{resolved.self_node_id}' timed out waiting for neighbors at round {round_idx}"
                )

            peer_payloads = [
                server.get_payload(neighbor.id, payload_id=payload_id) for neighbor in neighbors
            ]
            if any(payload is None for payload in peer_payloads):
                raise RuntimeError("Missing peer payload after successful barrier wait")

            peer_payloads = list(peer_payloads)
            merged_state = _merge_payloads(
                averaging=config.sync.averaging,
                base_state=base_state,
                local_payload=local_payload,
                peer_payloads=peer_payloads,
            )
            load_model_state(model, merged_state, device)

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

            state_digest = digest_state(extract_model_state(model))
            round_row = {
                "round": round_idx + 1,
                "local_train_loss": train_metrics["loss"],
                "local_train_accuracy": train_metrics["accuracy"],
                "local_train_macro_f1": train_metrics["macro_f1"],
                "samples_processed": train_metrics["samples_processed"],
                "duration_s": time.perf_counter() - round_started_perf,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_loss": final_test_metrics["loss"],
                "test_accuracy": final_test_metrics["accuracy"],
                "test_macro_f1": final_test_metrics["macro_f1"],
                "state_digest": state_digest,
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
                    checkpoint_dir / f"round-{round_idx + 1:03d}.pt",
                )

        metrics_ext = "json" if config.logging.metrics_format == "json" else "csv"
        write_metrics(log_dir / f"sync_round_metrics.{metrics_ext}", round_history, config.logging.metrics_format)
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
            "metrics_file": f"sync_round_metrics.{metrics_ext}",
            "scheduler_history_file": (
                str(scheduler_history_path) if scheduler_history_path is not None else None
            ),
            "transport_address": server.address,
            "advertise_address": _node_target(resolved.self_node.host, resolved.self_node.port),
            "neighbor_ids": [neighbor.id for neighbor in neighbors],
            "rounds": round_history,
            "final_test_metrics": final_test_metrics,
            "final_state_digest": round_history[-1]["state_digest"] if round_history else None,
            "data_plane_stats": data_plane_stats,
        }
        write_summary(log_dir / "sync_run_summary.json", summary)
        return summary
    finally:
        if adaptive_runtime is not None:
            adaptive_runtime.close()
        server.stop(grace=0.0)


def run_sync_smoke(
    peer_count: int = 3,
    rounds: int = 1,
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
        cluster_path, training_paths, node_ids = _write_sync_smoke_configs(
            tmp_path=tmp_path,
            peer_count=peer_count,
            storage_mode=storage_mode,
            scheduler_mode=scheduler_mode,
        )

        try:
            for node_id in node_ids:
                process = ctx.Process(
                    target=_sync_worker_process,
                    args=(
                        cluster_path,
                        training_paths[node_id],
                        node_id,
                        results_queue,
                        SyncRunOverrides(
                            rounds=rounds,
                            max_local_batches=2,
                            max_eval_batches=1,
                            run_name="sync-smoke",
                            transport_timeout_s=20.0,
                            bind_host="127.0.0.1",
                        ),
                    ),
                )
                process.start()
                processes.append(process)

            results = {}
            for _ in node_ids:
                item = results_queue.get(timeout=60.0)
                results[item["self_node_id"]] = item

            exit_codes = []
            for process in processes:
                process.join(timeout=20.0)
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


def _sync_worker_process(
    cluster_path: Path,
    training_path: Path,
    self_node_id: str,
    results_queue: mp.Queue,
    overrides: SyncRunOverrides,
) -> None:
    resolved = load_resolved_config(cluster_path, training_path, self_node_id)
    result = run_sync_worker(resolved, overrides=overrides)
    results_queue.put(result)


def _write_sync_smoke_configs(
    tmp_path: Path,
    peer_count: int,
    *,
    storage_mode: str = "replicated",
    scheduler_mode: str = "static",
) -> tuple[Path, dict[str, Path], list[str]]:
    node_ids = [f"node-{idx + 1}" for idx in range(peer_count)]
    ports = [_find_free_port() for _ in range(peer_count)]
    cluster_payload = {
        "cluster_name": "sync-smoke",
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
    training_payload["mode"] = "sync"
    training_payload["sync"]["enabled"] = True
    training_payload["sync"]["averaging"] = "deltas"
    training_payload["dataset"]["partitioning"] = "homogeneous"
    training_payload["dataset"]["scheduler_mode"] = scheduler_mode
    training_payload["dataset"]["rebalance_window_batches"] = 2
    training_payload["dataset"]["throughput_ema"] = 0.0
    training_payload["dataset"]["warmup_windows"] = 1
    training_payload["dataset"]["min_local_shards"] = 1
    training_payload["logging"]["log_dir"] = str(tmp_path / "logs")
    training_payload["logging"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    training_payload["optimization"]["eval_every_epochs"] = 1

    cluster_path = tmp_path / "cluster.sync-smoke.yaml"
    with cluster_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cluster_payload, handle, sort_keys=False)

    training_paths: dict[str, Path] = {}
    if storage_mode == "replicated":
        training_path = tmp_path / "training.sync-smoke.yaml"
        with training_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(training_payload, handle, sort_keys=False)
        for node_id in node_ids:
            training_paths[node_id] = training_path
        return cluster_path, training_paths, node_ids

    if storage_mode != "micro_shards":
        raise ValueError(f"Unsupported storage_mode for sync smoke: {storage_mode}")

    training_payload["dataset"]["storage_mode"] = "micro_shards"
    training_payload["dataset"]["shard_samples"] = 4
    bootstrap_manifest_path = tmp_path / node_ids[0] / "manifest.json"
    training_payload["dataset"]["manifest_path"] = str(bootstrap_manifest_path)
    training_payload["dataset"]["cache_dir"] = str(bootstrap_manifest_path.parent)
    bootstrap_training_path = tmp_path / node_ids[0] / "training.sync-smoke.micro.yaml"
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
        node_training_path = node_dir / "training.sync-smoke.micro.yaml"
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


def _train_for_steps(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    max_batches: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    processed_batches = 0
    confusion = init_confusion_matrix(num_classes)

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
        update_confusion_matrix(confusion, logits, targets)
        processed_batches += 1

        if processed_batches >= max_batches:
            break

    return summarize_classification_metrics(
        total_loss=total_loss,
        correct=correct,
        total=total,
        confusion=confusion,
    )


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


def _build_payload_state(
    *, averaging: str, current_state: dict[str, torch.Tensor], base_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if averaging == "weights":
        return current_state
    if averaging == "deltas":
        return compute_model_delta(current_state=current_state, base_state=base_state)
    raise ValueError(f"Unsupported sync averaging strategy: {averaging}")


def _merge_payloads(
    *,
    averaging: str,
    base_state: dict[str, torch.Tensor],
    local_payload: PeerPayload,
    peer_payloads: list[PeerPayload],
) -> dict[str, torch.Tensor]:
    all_payloads = [local_payload] + peer_payloads
    all_payloads.sort(key=lambda payload: payload.metadata.sender_node_id)
    states = [payload.tensors for payload in all_payloads]
    weights = [payload.metadata.sample_count for payload in all_payloads]
    averaged = weighted_average_states(states=states, weights=weights)
    if averaging == "weights":
        return averaged
    if averaging == "deltas":
        return add_state_delta(base_state=base_state, delta_state=averaged)
    raise ValueError(f"Unsupported sync averaging strategy: {averaging}")


def _node_target(host: str, port: int) -> str:
    return f"{host}:{port}"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
