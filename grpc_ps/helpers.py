import numpy as np
from typing import Iterable
import torch
from proto.ps_pb2 import TensorChunk

CHUNK_SIZE_BYTES = 1 * 1024 * 1024


def tensor_to_chunks(
    t: torch.Tensor, chunk_size: int = CHUNK_SIZE_BYTES
) -> Iterable[TensorChunk]:
    with torch.no_grad():
        array = t.contiguous().numpy()
    buf = memoryview(array.tobytes())
    num_chunks = max(1, (len(buf) + chunk_size - 1) // chunk_size)

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, len(buf))
        yield TensorChunk(
            data=buf[start:end].tobytes(),
            dtype=str(array.dtype),
            shape=t.shape,
            is_last=(chunk_id == num_chunks - 1),
        )


def chunks_to_tensor(chunks: list[TensorChunk]):
    if not chunks:
        raise ValueError("No chunks to assemble")
    buffer = b"".join(chunk.data for chunk in chunks)
    array = np.frombuffer(buffer, dtype=chunks[0].dtype)
    tensor = torch.from_numpy(array.copy()).reshape(tuple(chunks[0].shape))
    return tensor
