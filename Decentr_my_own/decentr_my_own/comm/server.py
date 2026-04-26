from __future__ import annotations

from concurrent import futures
from pathlib import Path

import grpc

from decentr_my_own.comm.rpc_conversion import (
    lease_plan_to_proto,
    manifest_to_proto,
    run_completion_from_proto,
    run_completion_to_proto,
    shard_meta_to_proto,
    throughput_report_from_proto,
)
from decentr_my_own.comm.proto import peer_pb2, peer_pb2_grpc
from decentr_my_own.comm.serialization import messages_to_payload
from decentr_my_own.comm.state import (
    ControlPlaneSnapshot,
    ControlPlaneStateStore,
    PeerStateSnapshot,
    PeerStateStore,
)
from decentr_my_own.data.shard_store import ShardStore
from decentr_my_own.data.shard_transfer import iter_shard_file_chunks


GRPC_OPTIONS = [
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
]
GRPC_COMPRESSION = grpc.Compression.Gzip


class PeerTransportServicer(peer_pb2_grpc.PeerTransportServicer):
    def __init__(
        self,
        node_id: str,
        state_store: PeerStateStore,
        control_store: ControlPlaneStateStore,
        shard_store: ShardStore | None = None,
        *,
        transfer_chunk_bytes: int = 1_048_576,
    ):
        self.node_id = node_id
        self.state_store = state_store
        self.control_store = control_store
        self.shard_store = shard_store
        self.transfer_chunk_bytes = transfer_chunk_bytes

    def Ping(self, request, context):
        return peer_pb2.PingReply(
            receiver_node_id=self.node_id,
            message=f"pong:{request.sender_node_id}->{self.node_id}",
        )

    def GetManifest(self, request, context):
        shard_store = self._require_shard_store(context)
        if shard_store is None:
            return peer_pb2.ManifestReply()
        return peer_pb2.ManifestReply(
            receiver_node_id=self.node_id,
            manifest=manifest_to_proto(shard_store.manifest),
        )

    def ListLocalShards(self, request, context):
        shard_store = self._require_shard_store(context)
        if shard_store is None:
            return peer_pb2.LocalShardsReply()
        return peer_pb2.LocalShardsReply(
            receiver_node_id=self.node_id,
            shards=[shard_meta_to_proto(shard) for shard in shard_store.list_local_shards()],
        )

    def PullShard(self, request, context):
        shard_store = self._require_shard_store(context)
        if shard_store is None:
            return
        try:
            shard_meta = shard_store.get_shard_meta(request.shard_id)
        except KeyError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        shard_path = shard_store.resolve_path(shard_meta)
        for offset, data, eof in iter_shard_file_chunks(
            shard_path,
            shard_id=shard_meta.shard_id,
            chunk_size=self.transfer_chunk_bytes,
        ):
            yield peer_pb2.ShardChunk(
                shard_id=shard_meta.shard_id,
                offset=offset,
                data=data,
                eof=eof,
            )

    def ReportThroughput(self, request, context):
        report = throughput_report_from_proto(request)
        stored = self.control_store.store_throughput_report(report)
        return peer_pb2.ReportThroughputReply(
            receiver_node_id=self.node_id,
            node_id=stored.node_id,
            window_id=stored.window_id,
        )

    def GetLeasePlan(self, request, context):
        lease_plan = self.control_store.get_lease_plan(request.window_id)
        return peer_pb2.LeasePlanReply(
            receiver_node_id=self.node_id,
            lease_plan=lease_plan_to_proto(lease_plan),
        )

    def ReportRunCompletion(self, request, context):
        completion = run_completion_from_proto(request)
        stored = self.control_store.store_run_completion(completion)
        return peer_pb2.ReportRunCompletionReply(
            receiver_node_id=self.node_id,
            node_id=stored.node_id,
        )

    def GetRunCompletions(self, request, context):
        return peer_pb2.RunCompletionsReply(
            receiver_node_id=self.node_id,
            completions=[
                run_completion_to_proto(item)
                for item in self.control_store.get_run_completions()
            ],
        )

    def PushPayload(self, request_iterator, context):
        payload, num_bytes, digest = messages_to_payload(request_iterator)
        summary = self.state_store.store(payload, num_bytes=num_bytes, digest=digest)
        return peer_pb2.PushPayloadReply(
            receiver_node_id=self.node_id,
            sender_node_id=summary.sender_node_id,
            payload_id=summary.payload_id,
            tensor_count=summary.tensor_count,
            num_bytes=summary.num_bytes,
            digest=summary.digest,
        )

    def GetPeerState(self, request, context):
        snapshot = self.state_store.snapshot()
        return _snapshot_to_reply(snapshot)

    def _require_shard_store(self, context) -> ShardStore | None:
        if self.shard_store is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Node '{self.node_id}' is not configured for shard transfer",
            )
        return self.shard_store


class PeerServer:
    def __init__(
        self,
        *,
        node_id: str,
        host: str = "127.0.0.1",
        port: int = 0,
        max_workers: int = 8,
        shard_manifest_path: str | Path | None = None,
        transfer_chunk_bytes: int = 1_048_576,
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.state_store = PeerStateStore(receiver_node_id=node_id)
        self.control_store = ControlPlaneStateStore(receiver_node_id=node_id)
        self.shard_store = (
            None
            if shard_manifest_path is None
            else ShardStore.from_manifest_path(shard_manifest_path)
        )
        self.servicer = PeerTransportServicer(
            node_id=node_id,
            state_store=self.state_store,
            control_store=self.control_store,
            shard_store=self.shard_store,
            transfer_chunk_bytes=transfer_chunk_bytes,
        )
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=GRPC_OPTIONS,
            compression=GRPC_COMPRESSION,
        )
        peer_pb2_grpc.add_PeerTransportServicer_to_server(
            self.servicer,
            self._server,
        )
        self.bound_port = self._server.add_insecure_port(f"{host}:{port}")
        if self.bound_port <= 0:
            raise RuntimeError(f"Failed to bind gRPC server for node '{node_id}'")

    @property
    def address(self) -> str:
        return f"{self.host}:{self.bound_port}"

    def start(self) -> None:
        self._server.start()

    def stop(self, grace: float = 0.0) -> None:
        self._server.stop(grace)

    def wait_for_termination(self, timeout: float | None = None) -> bool:
        return self._server.wait_for_termination(timeout=timeout)

    def snapshot(self) -> PeerStateSnapshot:
        return self.state_store.snapshot()

    def control_snapshot(self) -> ControlPlaneSnapshot:
        return self.control_store.snapshot()

    def get_payload(self, sender_node_id: str, payload_id: str | None = None):
        return self.state_store.get_payload(sender_node_id=sender_node_id, payload_id=payload_id)

    def wait_for_payloads(self, min_count: int, timeout_s: float) -> bool:
        return self.state_store.wait_for_payloads(min_count=min_count, timeout_s=timeout_s)

    def wait_for_payload_ids(
        self, sender_node_ids: list[str], payload_id: str, timeout_s: float
    ) -> bool:
        return self.state_store.wait_for_payload_ids(
            sender_node_ids=sender_node_ids,
            payload_id=payload_id,
            timeout_s=timeout_s,
        )

    def set_lease_plan(self, lease_plan) -> None:
        self.control_store.set_lease_plan(lease_plan)

    def configure_shard_store(
        self,
        manifest_path: str | Path,
        *,
        base_dir: str | Path | None = None,
    ) -> None:
        self.shard_store = ShardStore.from_manifest_path(manifest_path, base_dir=base_dir)
        self.servicer.shard_store = self.shard_store

    def get_throughput_reports(self, *, window_id: int | None = None):
        return self.control_store.get_throughput_reports(window_id=window_id)

    def wait_for_throughput_reports(
        self,
        *,
        node_ids: list[str],
        window_id: int,
        timeout_s: float,
    ) -> bool:
        return self.control_store.wait_for_throughput_reports(
            node_ids=node_ids,
            window_id=window_id,
            timeout_s=timeout_s,
        )

    def store_run_completion(self, completion) -> None:
        self.control_store.store_run_completion(completion)

    def get_run_completions(self):
        return self.control_store.get_run_completions()

    def wait_for_run_completions(self, *, node_ids: list[str], timeout_s: float) -> bool:
        return self.control_store.wait_for_run_completions(
            node_ids=node_ids,
            timeout_s=timeout_s,
        )


def _snapshot_to_reply(snapshot: PeerStateSnapshot) -> peer_pb2.PeerStateReply:
    return peer_pb2.PeerStateReply(
        receiver_node_id=snapshot.node_id,
        received_payload_count=snapshot.received_payload_count,
        payloads=[
            peer_pb2.StoredPayloadSummary(
                receiver_node_id=item.receiver_node_id,
                sender_node_id=item.sender_node_id,
                payload_id=item.payload_id,
                payload_kind=item.payload_kind,
                model_version=item.model_version,
                step=item.step,
                sample_count=item.sample_count,
                tensor_count=item.tensor_count,
                num_bytes=item.num_bytes,
                digest=item.digest,
                received_at=item.received_at,
                tensor_names=item.tensor_names,
            )
            for item in snapshot.payloads
        ],
    )
