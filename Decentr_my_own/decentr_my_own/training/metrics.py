from __future__ import annotations

import torch


def init_confusion_matrix(num_classes: int) -> torch.Tensor:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    return torch.zeros((num_classes, num_classes), dtype=torch.int64)


def update_confusion_matrix(
    confusion: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    num_classes = confusion.size(0)
    preds = logits.argmax(dim=1).detach().to("cpu", dtype=torch.int64)
    targets_cpu = targets.detach().to("cpu", dtype=torch.int64)
    bincount = torch.bincount(
        targets_cpu * num_classes + preds,
        minlength=num_classes * num_classes,
    )
    confusion += bincount.reshape(num_classes, num_classes)


def macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    true_positive = confusion.diag().to(torch.float64)
    false_positive = confusion.sum(dim=0).to(torch.float64) - true_positive
    false_negative = confusion.sum(dim=1).to(torch.float64) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    f1 = torch.where(
        denominator > 0,
        (2.0 * true_positive) / denominator,
        torch.zeros_like(denominator),
    )
    return float(f1.mean().item())


def summarize_classification_metrics(
    *,
    total_loss: float,
    correct: int,
    total: int,
    confusion: torch.Tensor,
) -> dict[str, float]:
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1_from_confusion(confusion),
        "samples_processed": total,
    }
