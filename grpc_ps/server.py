import threading
import grpc
import torch
from torch import optim
from torch.utils.data import DataLoader

import proto.ps_pb2_grpc as ps_grpc
from proto.ps_pb2 import UpdateRequest, UpdateResponse, TensorProto

import numpy as np
from helpers import serialize_tensor, deserialize_tensor


class ParameterServerServicer(ps_grpc.ParameterServerServicer):
    def __init__(
        self,
        model: torch.nn.Module,
        world_size: int,
        val_loader: DataLoader,
    ):
        self.model = model
        self.world_size = world_size
        self.lock = threading.Lock()
        self.val_loader = val_loader
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.9,
            # weight_decay=weight_decay,
            nesterov=True,
        )

    def SyncUpdate(self, request: UpdateRequest, context: grpc.ServicerContext):
        raise NotImplementedError("kaboom")

    def AsyncUpdate(self, request: UpdateRequest, context: grpc.ServicerContext):
        grads = [
            deserialize_tensor(
                tensor_proto.data,
                tuple(tensor_proto.shape),
                np.dtype(tensor_proto.dtype),
            )
            for tensor_proto in request.gradients
        ]

        with self.lock:
            with torch.no_grad():
                for p, g in zip(self.model.parameters(), grads):
                    p.grad = g

            self.optimizer.step()
            self.optimizer.zero_grad()

            return self._make_update_response()

    def _make_update_response(self) -> UpdateResponse:
        return UpdateResponse(
            parameters=[
                TensorProto(
                    *serialize_tensor(p),
                    shape=p.shape,
                )
                for p in self.model.parameters()
            ]
        )
