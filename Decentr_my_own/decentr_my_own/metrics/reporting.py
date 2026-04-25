from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from decentr_my_own.algorithms.async_gossip import run_async_smoke
from decentr_my_own.algorithms.sync_barrier import run_sync_smoke


SUMMARY_FILENAMES = (
    "run_summary.json",
    "sync_run_summary.json",
    "async_run_summary.json",
)


def build_run_report(log_root: str | Path, run_id: str) -> dict:
    log_root_path = Path(log_root)
    node_summaries = []
    for summary_path in discover_run_summary_paths(log_root_path, run_id):
        node_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    if not node_summaries:
        raise FileNotFoundError(
            f"No run summaries found under '{log_root_path / run_id}'."
        )
    return aggregate_node_summaries(node_summaries, run_id=run_id)


def compare_run_reports(baseline: dict, candidate: dict) -> dict:
    deltas = {}
    for key in (
        "total_samples_processed",
        "effective_samples_per_s_total",
        "effective_samples_per_s_mean",
        "run_duration_s_max",
        "final_test_accuracy_mean",
        "final_test_loss_mean",
        "best_val_accuracy_mean",
    ):
        deltas[key] = _subtract(candidate.get(key), baseline.get(key))

    ratios = {
        "throughput_total_ratio": _ratio(
            candidate.get("effective_samples_per_s_total"),
            baseline.get("effective_samples_per_s_total"),
        ),
        "duration_max_ratio": _ratio(
            candidate.get("run_duration_s_max"),
            baseline.get("run_duration_s_max"),
        ),
        "final_test_accuracy_ratio": _ratio(
            candidate.get("final_test_accuracy_mean"),
            baseline.get("final_test_accuracy_mean"),
        ),
    }

    return {
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "baseline_modes": baseline.get("modes", []),
        "candidate_modes": candidate.get("modes", []),
        "deltas": deltas,
        "ratios": ratios,
        "winner": {
            "accuracy": _winner_higher(
                baseline.get("final_test_accuracy_mean"),
                candidate.get("final_test_accuracy_mean"),
                baseline.get("run_id"),
                candidate.get("run_id"),
            ),
            "throughput": _winner_higher(
                baseline.get("effective_samples_per_s_total"),
                candidate.get("effective_samples_per_s_total"),
                baseline.get("run_id"),
                candidate.get("run_id"),
            ),
            "duration": _winner_lower(
                baseline.get("run_duration_s_max"),
                candidate.get("run_duration_s_max"),
                baseline.get("run_id"),
                candidate.get("run_id"),
            ),
        },
    }


def build_smoke_comparison(
    *,
    peer_count: int = 3,
    sync_rounds: int = 1,
    async_rounds: int = 2,
) -> dict:
    sync_result = run_sync_smoke(peer_count=peer_count, rounds=sync_rounds)
    async_result = run_async_smoke(peer_count=peer_count, rounds=async_rounds)

    sync_report = aggregate_node_summaries(
        sync_result["node_results"].values(),
        run_id=f"sync-smoke-{peer_count}p-{sync_rounds}r",
    )
    async_report = aggregate_node_summaries(
        async_result["node_results"].values(),
        run_id=f"async-smoke-{peer_count}p-{async_rounds}r",
    )
    return {
        "sync": sync_report,
        "async": async_report,
        "comparison": compare_run_reports(sync_report, async_report),
    }


def aggregate_node_summaries(
    node_summaries: Iterable[dict],
    *,
    run_id: str | None = None,
) -> dict:
    summaries = list(node_summaries)
    if not summaries:
        raise ValueError("node_summaries must not be empty")

    node_rows = []
    durations = []
    throughputs = []
    final_acc = []
    final_loss = []
    best_val_acc = []
    samples = []
    state_digests = []
    mixed_peer_updates = []
    max_staleness = []
    failed_pushes = []

    for item in summaries:
        final_metrics = item.get("final_test_metrics", {})
        node_duration = _float_or_none(item.get("run_duration_s"))
        node_throughput = _float_or_none(item.get("effective_samples_per_s"))
        node_accuracy = _float_or_none(final_metrics.get("accuracy"))
        node_loss = _float_or_none(final_metrics.get("loss"))
        node_best_val = _float_or_none(item.get("best_val_accuracy"))
        node_samples = _int_or_none(item.get("total_samples_processed"))
        node_digest = item.get("final_state_digest")

        durations.append(node_duration)
        throughputs.append(node_throughput)
        final_acc.append(node_accuracy)
        final_loss.append(node_loss)
        best_val_acc.append(node_best_val)
        samples.append(node_samples)
        state_digests.append(node_digest)

        if item.get("mixed_peer_updates_total") is not None:
            mixed_peer_updates.append(_int_or_none(item.get("mixed_peer_updates_total")))
        if item.get("max_observed_staleness") is not None:
            max_staleness.append(_int_or_none(item.get("max_observed_staleness")))
        if item.get("failed_pushes_total") is not None:
            failed_pushes.append(_int_or_none(item.get("failed_pushes_total")))

        node_rows.append(
            {
                "self_node_id": item.get("self_node_id"),
                "device": item.get("device"),
                "mode": item.get("mode"),
                "algorithm": item.get("algorithm"),
                "run_duration_s": node_duration,
                "effective_samples_per_s": node_throughput,
                "total_samples_processed": node_samples,
                "final_test_accuracy": node_accuracy,
                "final_test_loss": node_loss,
                "best_val_accuracy": node_best_val,
                "final_state_digest": node_digest,
            }
        )

    unique_digests = {digest for digest in state_digests if digest}
    report = {
        "run_id": run_id or summaries[0].get("run_id"),
        "cluster_names": sorted({item.get("cluster_name") for item in summaries if item.get("cluster_name")}),
        "node_count": len(summaries),
        "modes": sorted({item.get("mode") for item in summaries if item.get("mode")}),
        "algorithms": sorted(
            {item.get("algorithm") for item in summaries if item.get("algorithm")}
        ),
        "datasets": sorted({item.get("dataset") for item in summaries if item.get("dataset")}),
        "models": sorted({item.get("model") for item in summaries if item.get("model")}),
        "history_kinds": sorted(
            {item.get("history_kind") for item in summaries if item.get("history_kind")}
        ),
        "total_samples_processed": int(sum(value for value in samples if value is not None)),
        "effective_samples_per_s_total": round(
            sum(value for value in throughputs if value is not None), 6
        ),
        "effective_samples_per_s_mean": _mean(throughputs),
        "run_duration_s_max": _max(durations),
        "run_duration_s_mean": _mean(durations),
        "final_test_accuracy_mean": _mean(final_acc),
        "final_test_accuracy_max": _max(final_acc),
        "final_test_loss_mean": _mean(final_loss),
        "best_val_accuracy_mean": _mean(best_val_acc),
        "best_val_accuracy_max": _max(best_val_acc),
        "consistent_final_state": len(unique_digests) == 1 and len(unique_digests) > 0,
        "nodes": sorted(node_rows, key=lambda item: item["self_node_id"] or ""),
    }
    if mixed_peer_updates:
        report["mixed_peer_updates_total_sum"] = int(
            sum(value for value in mixed_peer_updates if value is not None)
        )
    if max_staleness:
        report["max_observed_staleness_max"] = max(
            value for value in max_staleness if value is not None
        )
    if failed_pushes:
        report["failed_pushes_total_sum"] = int(
            sum(value for value in failed_pushes if value is not None)
        )
    return report


def discover_run_summary_paths(log_root: Path, run_id: str) -> list[Path]:
    run_path = log_root / run_id
    if not run_path.exists():
        return []

    paths = []
    for node_dir in sorted(path for path in run_path.iterdir() if path.is_dir()):
        for filename in SUMMARY_FILENAMES:
            summary_path = node_dir / filename
            if summary_path.exists():
                paths.append(summary_path)
                break
    return paths


def _mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 6)


def _max(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(max(filtered), 6)


def _subtract(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _ratio(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right in (None, 0):
        return None
    return round(float(left) / float(right), 6)


def _winner_higher(
    baseline_value: float | None,
    candidate_value: float | None,
    baseline_label: str | None,
    candidate_label: str | None,
) -> str | None:
    if baseline_value is None or candidate_value is None:
        return None
    if candidate_value > baseline_value:
        return candidate_label
    if candidate_value < baseline_value:
        return baseline_label
    return "tie"


def _winner_lower(
    baseline_value: float | None,
    candidate_value: float | None,
    baseline_label: str | None,
    candidate_label: str | None,
) -> str | None:
    if baseline_value is None or candidate_value is None:
        return None
    if candidate_value < baseline_value:
        return candidate_label
    if candidate_value > baseline_value:
        return baseline_label
    return "tie"


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    return int(value)
