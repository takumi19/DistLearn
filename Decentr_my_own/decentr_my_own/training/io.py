from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def create_run_directories(
    log_dir: str,
    checkpoint_dir: str,
    node_id: str,
    run_name: str | None = None,
) -> tuple[str, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = run_name or f"{node_id}-{timestamp}"
    log_path = Path(log_dir) / run_id / node_id
    checkpoint_path = Path(checkpoint_dir) / run_id / node_id
    log_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return run_id, log_path, checkpoint_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_rate(count: int | float, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return float(count) / duration_s


def write_metrics(path: Path, rows: Iterable[dict], fmt: str) -> None:
    rows = list(rows)
    if fmt == "json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
