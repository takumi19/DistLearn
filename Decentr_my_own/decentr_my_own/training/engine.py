from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from decentr_my_own.config.models import ResolvedConfig
from decentr_my_own.data.loaders import build_local_dataloaders
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


@dataclass
class LocalTrainOverrides:
    epochs: int | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str | None = None


@dataclass
class LocalTrainResult:
    run_id: str
    device: str
    log_dir: Path
    checkpoint_dir: Path
    epoch_history: list[dict]
    final_test_metrics: dict

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "device": self.device,
            "log_dir": str(self.log_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "epochs": self.epoch_history,
            "final_test_metrics": self.final_test_metrics,
        }


def run_local_training(
    resolved: ResolvedConfig, overrides: LocalTrainOverrides | None = None
) -> LocalTrainResult:
    overrides = overrides or LocalTrainOverrides()
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

    dataloaders = build_local_dataloaders(resolved, pin_memory=pin_memory)
    model = build_model(config.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.optimization.lr,
        momentum=config.optimization.momentum,
        weight_decay=config.optimization.weight_decay,
    )

    epoch_count = overrides.epochs or config.optimization.epochs
    epoch_history: list[dict] = []
    final_test_metrics = {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

    for epoch_idx in range(epoch_count):
        epoch_started_perf = time.perf_counter()
        if dataloaders.train_sampler is not None:
            dataloaders.train_sampler.set_epoch(epoch_idx)

        train_metrics = _run_train_epoch(
            model,
            dataloaders.train,
            optimizer,
            criterion,
            device,
            num_classes=config.model.num_classes,
            max_batches=overrides.max_train_batches,
        )

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

        epoch_row = {
            "epoch": epoch_idx + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "samples_processed": train_metrics["samples_processed"],
            "duration_s": time.perf_counter() - epoch_started_perf,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_loss": final_test_metrics["loss"],
            "test_accuracy": final_test_metrics["accuracy"],
            "test_macro_f1": final_test_metrics["macro_f1"],
        }
        epoch_row["samples_per_s"] = safe_rate(
            epoch_row["samples_processed"], epoch_row["duration_s"]
        )
        epoch_history.append(epoch_row)

        if (epoch_idx + 1) % config.logging.save_every_round == 0:
            checkpoint_payload = {
                "epoch": epoch_idx + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": epoch_row,
                "self_node_id": resolved.self_node_id,
            }
            torch.save(
                checkpoint_payload,
                checkpoint_dir / f"epoch-{epoch_idx + 1:03d}.pt",
            )

    metrics_ext = "json" if config.logging.metrics_format == "json" else "csv"
    write_metrics(log_dir / f"epoch_metrics.{metrics_ext}", epoch_history, config.logging.metrics_format)
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
    best_test_accuracy = max((row["test_accuracy"] for row in epoch_history), default=None)
    best_val_macro_f1 = max(
        (
            row["val_macro_f1"]
            for row in epoch_history
            if row.get("val_macro_f1") is not None
        ),
        default=None,
    )
    best_test_macro_f1 = max((row["test_macro_f1"] for row in epoch_history), default=None)
    summary = {
        "run_id": run_id,
        "cluster_name": resolved.cluster.cluster_name,
        "self_node_id": resolved.self_node_id,
        "cluster_node_count": len(resolved.cluster.nodes),
        "device": str(device),
        "dataset": config.dataset.name,
        "model": config.model.name,
        "mode": config.mode,
        "algorithm": config.algorithm,
        "history_kind": "epochs",
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "run_duration_s": run_duration_s,
        "epoch_count": len(epoch_history),
        "train_sample_count": dataloaders.train_sample_count,
        "total_samples_processed": total_samples_processed,
        "effective_samples_per_s": safe_rate(total_samples_processed, run_duration_s),
        "best_val_accuracy": best_val_accuracy,
        "best_test_accuracy": best_test_accuracy,
        "best_val_macro_f1": best_val_macro_f1,
        "best_test_macro_f1": best_test_macro_f1,
        "metrics_file": f"epoch_metrics.{metrics_ext}",
        "epochs": epoch_history,
        "final_test_metrics": final_test_metrics,
    }
    write_summary(log_dir / "run_summary.json", summary)

    return LocalTrainResult(
        run_id=run_id,
        device=str(device),
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        epoch_history=epoch_history,
        final_test_metrics=final_test_metrics,
    )


def _run_train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = init_confusion_matrix(num_classes)

    for batch_idx, (inputs, targets) in enumerate(loader):
        targets_cpu = targets.detach().to("cpu", dtype=torch.int64).reshape(-1)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets_cpu.numel()
        total_loss += loss.item() * batch_size
        preds_cpu = logits.argmax(dim=1).detach().to("cpu", dtype=torch.int64).reshape(-1)
        correct += (preds_cpu == targets_cpu).sum().item()
        total += batch_size
        update_confusion_matrix(confusion, logits, targets_cpu)

        if max_batches is not None and batch_idx + 1 >= max_batches:
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
    confusion = init_confusion_matrix(num_classes)

    for batch_idx, (inputs, targets) in enumerate(loader):
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

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    metrics = summarize_classification_metrics(
        total_loss=total_loss,
        correct=correct,
        total=total,
        confusion=confusion,
    )
    metrics.pop("samples_processed", None)
    return metrics
