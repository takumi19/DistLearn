from __future__ import annotations

from decentr_my_own.comm.proto import peer_pb2
from decentr_my_own.data.manifest import DatasetManifest, ShardMeta, SplitSummary
from decentr_my_own.data.scheduler_state import (
    LeasePlanRecord,
    RunCompletionRecord,
    ThroughputReportRecord,
)


def manifest_to_proto(manifest: DatasetManifest) -> peer_pb2.ManifestSnapshot:
    return peer_pb2.ManifestSnapshot(
        format_version=manifest.format_version,
        dataset_name=manifest.dataset_name,
        storage_mode=manifest.storage_mode,
        seed=manifest.seed,
        shard_samples=manifest.shard_samples,
        num_classes=manifest.num_classes,
        image_shape=list(manifest.image_shape),
        splits=[
            peer_pb2.SplitSummary(
                split=summary.split,
                sample_count=summary.sample_count,
                shard_count=summary.shard_count,
            )
            for _, summary in sorted(manifest.splits.items())
        ],
        shards=[shard_meta_to_proto(shard) for shard in manifest.shards],
    )


def manifest_from_proto(message: peer_pb2.ManifestSnapshot) -> DatasetManifest:
    return DatasetManifest(
        format_version=message.format_version,
        dataset_name=message.dataset_name,
        storage_mode=message.storage_mode,
        seed=message.seed,
        shard_samples=message.shard_samples,
        num_classes=message.num_classes,
        image_shape=tuple(int(item) for item in message.image_shape),
        splits={
            item.split: SplitSummary(
                split=item.split,
                sample_count=item.sample_count,
                shard_count=item.shard_count,
            )
            for item in message.splits
        },
        shards=[shard_meta_from_proto(item) for item in message.shards],
    )


def shard_meta_to_proto(shard: ShardMeta) -> peer_pb2.ShardMeta:
    return peer_pb2.ShardMeta(
        shard_id=shard.shard_id,
        split=shard.split,
        relative_path=shard.relative_path,
        sample_count=shard.sample_count,
        byte_size=shard.byte_size,
        sha256=shard.sha256,
    )


def shard_meta_from_proto(message: peer_pb2.ShardMeta) -> ShardMeta:
    return ShardMeta(
        shard_id=message.shard_id,
        split=message.split,
        relative_path=message.relative_path,
        sample_count=message.sample_count,
        byte_size=message.byte_size,
        sha256=message.sha256,
    )


def throughput_report_to_proto(report: ThroughputReportRecord) -> peer_pb2.ThroughputReport:
    return peer_pb2.ThroughputReport(
        node_id=report.node_id,
        window_id=report.window_id,
        samples_processed=report.samples_processed,
        window_seconds=report.window_seconds,
        effective_throughput=report.effective_throughput,
        local_inventory=list(report.local_inventory),
    )


def throughput_report_from_proto(message: peer_pb2.ThroughputReport) -> ThroughputReportRecord:
    return ThroughputReportRecord(
        node_id=message.node_id,
        window_id=message.window_id,
        samples_processed=message.samples_processed,
        window_seconds=message.window_seconds,
        effective_throughput=message.effective_throughput,
        local_inventory=tuple(message.local_inventory),
    )


def lease_plan_to_proto(plan: LeasePlanRecord) -> peer_pb2.LeasePlan:
    return peer_pb2.LeasePlan(
        window_id=plan.window_id,
        epoch_id=plan.epoch_id,
        epoch_window_index=plan.epoch_window_index,
        epoch_window_count=plan.epoch_window_count,
        assignments=[
            peer_pb2.LeaseAssignment(node_id=node_id, shard_ids=list(shard_ids))
            for node_id, shard_ids in sorted(plan.assignments.items())
        ],
    )


def lease_plan_from_proto(message: peer_pb2.LeasePlan) -> LeasePlanRecord:
    return LeasePlanRecord(
        window_id=message.window_id,
        epoch_id=message.epoch_id,
        epoch_window_index=message.epoch_window_index,
        epoch_window_count=message.epoch_window_count or 1,
        assignments={
            item.node_id: tuple(item.shard_ids) for item in message.assignments
        },
    )


def run_completion_to_proto(record: RunCompletionRecord) -> peer_pb2.RunCompletion:
    return peer_pb2.RunCompletion(
        node_id=record.node_id,
        last_window_id=record.last_window_id,
        total_samples_processed=record.total_samples_processed,
        final_state_digest=record.final_state_digest or "",
        completed_at=record.completed_at or "",
    )


def run_completion_from_proto(message: peer_pb2.RunCompletion) -> RunCompletionRecord:
    return RunCompletionRecord(
        node_id=message.node_id,
        last_window_id=message.last_window_id,
        total_samples_processed=message.total_samples_processed,
        final_state_digest=message.final_state_digest or None,
        completed_at=message.completed_at or None,
    )
