from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import grpc

from decentr_my_own.comm.rpc_conversion import (
    lease_plan_from_proto,
    manifest_from_proto,
    run_completion_from_proto,
    run_completion_to_proto,
    shard_meta_from_proto,
    throughput_report_to_proto,
)
from decentr_my_own.comm.messages import PeerPayload
from decentr_my_own.comm.proto import peer_pb2, peer_pb2_grpc
from decentr_my_own.comm.serialization import payload_to_messages
from decentr_my_own.comm.server import GRPC_COMPRESSION, GRPC_OPTIONS
from decentr_my_own.data.manifest import DatasetManifest, ShardMeta
from decentr_my_own.data.scheduler_state import (
    LeasePlanRecord,
    RunCompletionRecord,
    ThroughputReportRecord,
)
from decentr_my_own.data.shard_transfer import PulledShardResult, write_pulled_shard


@dataclass(frozen=True)
class PushResult:
    receiver_node_id: str
    sender_node_id: str
    payload_id: str
    tensor_count: int
    num_bytes: int
    digest: str

    def to_dict(self) -> dict:
        return {
            "receiver_node_id": self.receiver_node_id,
            "sender_node_id": self.sender_node_id,
            "payload_id": self.payload_id,
            "tensor_count": self.tensor_count,
            "num_bytes": self.num_bytes,
            "digest": self.digest,
        }


class PeerClient:
    def __init__(self, target: str):
        self.target = target
        self.channel = grpc.insecure_channel(
            target,
            options=GRPC_OPTIONS,
            compression=GRPC_COMPRESSION,
        )
        self.stub = peer_pb2_grpc.PeerTransportStub(self.channel)

    def close(self) -> None:
        self.channel.close()

    def ping(self, sender_node_id: str, timeout_s: float = 5.0) -> dict:
        reply = self.stub.Ping(
            peer_pb2.PingRequest(sender_node_id=sender_node_id),
            timeout=timeout_s,
        )
        return {
            "receiver_node_id": reply.receiver_node_id,
            "message": reply.message,
        }

    def get_manifest(self, timeout_s: float = 5.0) -> DatasetManifest:
        reply = self.stub.GetManifest(peer_pb2.ManifestRequest(), timeout=timeout_s)
        return manifest_from_proto(reply.manifest)

    def list_local_shards(self, timeout_s: float = 5.0) -> list[ShardMeta]:
        reply = self.stub.ListLocalShards(peer_pb2.LocalShardsRequest(), timeout=timeout_s)
        return [shard_meta_from_proto(item) for item in reply.shards]

    def pull_shard(
        self,
        *,
        shard_id: str,
        destination_path: str | Path,
        expected_meta: ShardMeta | None = None,
        timeout_s: float = 30.0,
    ) -> PulledShardResult:
        response_iterator = self.stub.PullShard(
            peer_pb2.PullShardRequest(shard_id=shard_id),
            timeout=timeout_s,
        )
        chunks: list[bytes] = []
        seen_eof = False
        expected_offset = 0
        for chunk in response_iterator:
            if chunk.shard_id != shard_id:
                raise ValueError(
                    f"Shard stream returned unexpected shard_id '{chunk.shard_id}' for '{shard_id}'"
                )
            if chunk.offset != expected_offset:
                raise ValueError(
                    f"Shard stream offset mismatch for '{shard_id}': "
                    f"expected {expected_offset}, got {chunk.offset}"
                )
            if chunk.eof:
                seen_eof = True
                break
            chunks.append(chunk.data)
            expected_offset += len(chunk.data)
        if not seen_eof:
            raise ValueError(f"Shard stream for '{shard_id}' ended without eof marker")
        return write_pulled_shard(
            shard_id=shard_id,
            destination_path=destination_path,
            chunks=chunks,
            expected_meta=expected_meta,
        )

    def report_throughput(
        self,
        report: ThroughputReportRecord,
        timeout_s: float = 5.0,
    ) -> dict:
        reply = self.stub.ReportThroughput(
            throughput_report_to_proto(report),
            timeout=timeout_s,
        )
        return {
            "receiver_node_id": reply.receiver_node_id,
            "node_id": reply.node_id,
            "window_id": reply.window_id,
        }

    def get_lease_plan(
        self,
        *,
        window_id: int = 0,
        timeout_s: float = 5.0,
    ) -> LeasePlanRecord:
        reply = self.stub.GetLeasePlan(
            peer_pb2.LeasePlanRequest(window_id=window_id),
            timeout=timeout_s,
        )
        return lease_plan_from_proto(reply.lease_plan)

    def report_run_completion(
        self,
        completion: RunCompletionRecord,
        timeout_s: float = 5.0,
    ) -> dict:
        reply = self.stub.ReportRunCompletion(
            run_completion_to_proto(completion),
            timeout=timeout_s,
        )
        return {
            "receiver_node_id": reply.receiver_node_id,
            "node_id": reply.node_id,
        }

    def get_run_completions(self, timeout_s: float = 5.0) -> list[RunCompletionRecord]:
        reply = self.stub.GetRunCompletions(
            peer_pb2.RunCompletionsRequest(),
            timeout=timeout_s,
        )
        return [run_completion_from_proto(item) for item in reply.completions]

    def push_payload(
        self, payload: PeerPayload, timeout_s: float = 15.0
    ) -> PushResult:
        reply = self.stub.PushPayload(
            payload_to_messages(payload),
            timeout=timeout_s,
            # Model weights are mostly random floats; gzip costs CPU on weak VPS
            # nodes while usually saving little bandwidth.
            compression=grpc.Compression.NoCompression,
        )
        return PushResult(
            receiver_node_id=reply.receiver_node_id,
            sender_node_id=reply.sender_node_id,
            payload_id=reply.payload_id,
            tensor_count=reply.tensor_count,
            num_bytes=reply.num_bytes,
            digest=reply.digest,
        )

    def get_peer_state(self, timeout_s: float = 5.0) -> dict:
        reply = self.stub.GetPeerState(peer_pb2.PeerStateRequest(), timeout=timeout_s)
        return {
            "receiver_node_id": reply.receiver_node_id,
            "received_payload_count": reply.received_payload_count,
            "payloads": [
                {
                    "receiver_node_id": item.receiver_node_id,
                    "sender_node_id": item.sender_node_id,
                    "payload_id": item.payload_id,
                    "payload_kind": item.payload_kind,
                    "model_version": item.model_version,
                    "step": item.step,
                    "sample_count": item.sample_count,
                    "tensor_count": item.tensor_count,
                    "num_bytes": item.num_bytes,
                    "digest": item.digest,
                    "received_at": item.received_at,
                    "tensor_names": list(item.tensor_names),
                }
                for item in reply.payloads
            ],
        }
