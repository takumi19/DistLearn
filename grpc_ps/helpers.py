import torch
import numpy as np


def serialize_tensor(t: torch.Tensor):
    array = t.contiguous().numpy()
    return array.tobytes(), str(array.dtype)


def deserialize_tensor(data: bytes, shape: tuple, dtype: np.dtype):
    array = np.frombuffer(data, dtype=dtype)
    tensor = torch.from_numpy(array).reshape(shape)
    return tensor
