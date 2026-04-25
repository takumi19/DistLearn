from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch

from decentr_my_own.comm.messages import PayloadMetadata, PeerPayload
from decentr_my_own.comm.proto import peer_pb2


CHUNK_SIZE_BYTES = 1 * 1024 * 1024


def payload_to_messages(
    payload: PeerPayload, chunk_size: int = CHUNK_SIZE_BYTES
) -> Iterable[peer_pb2.PayloadMessage]:
    tensor_names = list(payload.tensors.keys())
    yield peer_pb2.PayloadMessage(
        header=peer_pb2.PayloadHeader(
            sender_node_id=payload.metadata.sender_node_id,
            payload_id=payload.metadata.payload_id,
            payload_kind=payload.metadata.payload_kind,
            model_version=payload.metadata.model_version,
            step=payload.metadata.step,
            sample_count=payload.metadata.sample_count,
            tensor_names=tensor_names,
        )
    )

    for tensor_index, tensor_name in enumerate(tensor_names):
        yield from tensor_to_messages(
            payload.tensors[tensor_name], tensor_index=tensor_index, chunk_size=chunk_size
        )


def messages_to_payload(
    messages: Iterable[peer_pb2.PayloadMessage],
) -> tuple[PeerPayload, int, str]:
    header = None
    chunks_by_tensor: dict[int, list[peer_pb2.TensorChunk]] = defaultdict(list)
    digest = hashlib.sha256()
    num_bytes = 0

    for message in messages:
        kind = message.WhichOneof("item")
        if kind == "header":
            if header is not None:
                raise ValueError("Payload stream contains multiple headers")
            header = message.header
        elif kind == "tensor_chunk":
            chunk = message.tensor_chunk
            chunks_by_tensor[chunk.tensor_index].append(chunk)
            digest.update(chunk.data)
            num_bytes += len(chunk.data)
        else:
            raise ValueError("Payload stream contains an empty message")

    if header is None:
        raise ValueError("Payload stream is missing the header message")

    tensors: dict[str, torch.Tensor] = {}
    for tensor_index, tensor_name in enumerate(header.tensor_names):
        tensor_chunks = chunks_by_tensor.get(tensor_index, [])
        if not tensor_chunks:
            raise ValueError(f"Missing tensor chunks for tensor '{tensor_name}'")
        if not tensor_chunks[-1].is_last_chunk:
            raise ValueError(f"Tensor '{tensor_name}' stream is incomplete")
        tensors[tensor_name] = chunks_to_tensor(tensor_chunks)

    payload = PeerPayload(
        metadata=PayloadMetadata(
            sender_node_id=header.sender_node_id,
            payload_id=header.payload_id,
            payload_kind=header.payload_kind,
            model_version=header.model_version,
            step=header.step,
            sample_count=header.sample_count,
        ),
        tensors=tensors,
    )
    return payload, num_bytes, digest.hexdigest()


def tensor_to_messages(
    tensor: torch.Tensor, *, tensor_index: int, chunk_size: int = CHUNK_SIZE_BYTES
) -> Iterable[peer_pb2.PayloadMessage]:
    cpu_tensor = tensor.detach().to("cpu").contiguous()
    array = cpu_tensor.numpy()
    buf = memoryview(array.tobytes())
    num_chunks = max(1, (len(buf) + chunk_size - 1) // chunk_size)

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, len(buf))
        yield peer_pb2.PayloadMessage(
            tensor_chunk=peer_pb2.TensorChunk(
                tensor_index=tensor_index,
                data=buf[start:end].tobytes(),
                dtype=str(array.dtype),
                shape=cpu_tensor.shape,
                is_last_chunk=(chunk_id == num_chunks - 1),
            )
        )


def chunks_to_tensor(chunks: list[peer_pb2.TensorChunk]) -> torch.Tensor:
    if not chunks:
        raise ValueError("No chunks to reconstruct tensor")
    buffer = b"".join(chunk.data for chunk in chunks)
    array = np.frombuffer(buffer, dtype=chunks[0].dtype)
    return torch.from_numpy(array.copy()).reshape(tuple(chunks[0].shape))
