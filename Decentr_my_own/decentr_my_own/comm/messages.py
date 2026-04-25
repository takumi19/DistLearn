from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class PayloadMetadata:
    sender_node_id: str
    payload_id: str
    payload_kind: str
    model_version: int
    step: int
    sample_count: int


@dataclass
class PeerPayload:
    metadata: PayloadMetadata
    tensors: dict[str, torch.Tensor]


@dataclass(frozen=True)
class StoredPayloadSummary:
    receiver_node_id: str
    sender_node_id: str
    payload_id: str
    payload_kind: str
    model_version: int
    step: int
    sample_count: int
    tensor_count: int
    num_bytes: int
    digest: str
    received_at: str
    tensor_names: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "receiver_node_id": self.receiver_node_id,
            "sender_node_id": self.sender_node_id,
            "payload_id": self.payload_id,
            "payload_kind": self.payload_kind,
            "model_version": self.model_version,
            "step": self.step,
            "sample_count": self.sample_count,
            "tensor_count": self.tensor_count,
            "num_bytes": self.num_bytes,
            "digest": self.digest,
            "received_at": self.received_at,
            "tensor_names": list(self.tensor_names),
        }


@dataclass
class StoredPayload:
    summary: StoredPayloadSummary
    payload: PeerPayload
