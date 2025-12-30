import numpy as np
import torch
from proto.ps_pb2 import TensorProto


def serialize_tensor(tp: torch.Tensor) -> TensorProto:
    with torch.no_grad():
        array = tp.contiguous().numpy()
        return TensorProto(data=array.tobytes(), dtype=str(array.dtype), shape=tp.shape)


def deserialize_tensor(tp: TensorProto) -> torch.Tensor:
    array = np.frombuffer(tp.data, dtype=tp.dtype)
    tensor = torch.from_numpy(array.copy()).reshape(tuple(tp.shape))
    return tensor
