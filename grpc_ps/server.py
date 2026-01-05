from itertools import chain
import threading
from typing import Iterable
import time

import grpc
import proto.ps_pb2_grpc as ps_grpc
import torch
from helpers import (
    chunks_to_tensor,
    deserialize_tensor,
    serialize_tensor,
    tensor_to_chunks,
)
from proto.ps_pb2 import UpdateRequest, UpdateResponse, TensorChunk
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class ParameterServerServicer(ps_grpc.ParameterServerServicer):
    def __init__(
        self,
        model: torch.nn.Module,
        world_size: int,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
    ):
        self.model = model
        self.world_size = world_size
        self.lock = threading.Lock()
        self.val_loader = val_loader
        self.criterion = criterion
        self.aggregated_grads: list[list[torch.Tensor]] = []
        self.device = "cpu"
        self.sync_done = threading.Event()
        self.req_cnt = 0  # For the async function to run validation
        self.aggregated_buffers: list[list[torch.Tensor]] = []

    def StreamingSyncUpdate(
        self, request_iterator: Iterable[TensorChunk], context: grpc.ServicerContext
    ) -> Iterable[TensorChunk]:
        chunks: list[TensorChunk] = []
        param_updates: list[torch.Tensor] = []

        for chunk in request_iterator:
            chunks.append(chunk)
            if chunk.is_last:
                param_updates.append(chunks_to_tensor(chunks))
                chunks.clear()

        rank, epoch = None, None
        for k, v in context.invocation_metadata():
            if k == "rank":
                rank = int(v)
            elif k == "epoch":
                epoch = int(v)

        print(f"Received updated from {rank=}, for {epoch=}")

        with self.lock:
            self.aggregated_grads.append(param_updates)
            should_wait = len(self.aggregated_grads) != self.world_size - 1

        # NOTE: this block is not protected by a mutex but it will only be invoked by one worker
        # and only after the other calls are waiting on sync_done.
        if not should_wait:
            print(f"S[{rank}] updating params")
            avg_grads = [
                sum(g[i] for g in self.aggregated_grads) / (self.world_size - 1)
                for i in range(len(self.aggregated_grads[0]))
            ]
            print(f"S[{rank}] averaged params")

            params_and_bufs = chain(self.model.parameters(), self.model.buffers())
            with torch.no_grad():
                for p, g in zip(params_and_bufs, avg_grads):
                    p.copy_(p + g)
            print(f"S[{rank}] updated params")

            self.aggregated_grads = []

            thr = threading.Thread(
                name=f"Validation-{epoch}", target=self._run_validation
            )
            thr.start()

            print(f"S[{rank}] signals done")
            self.sync_done.set()
        else:
            print(f"S[{rank}] waiting")
            self.sync_done.wait()

        with self.lock:
            if self.sync_done.is_set():
                print(f"S[{rank}] clears condvar")
                self.sync_done.clear()

        print(f"S[{rank}] receiving chunks")
        yield from self._chunk_stream()

    def SyncUpdate(self, request: UpdateRequest, context: grpc.ServicerContext):
        with self.lock:
            print(f"Received updates from {request.rank} for epoch {request.epoch}")
            grads = [
                deserialize_tensor(tensor_proto) for tensor_proto in request.gradients
            ]
            self.aggregated_grads.append(grads)

            if len(self.aggregated_grads) == self.world_size - 1:
                avg_grads = [
                    sum(g[i] for g in self.aggregated_grads) / (self.world_size - 1)
                    for i in range(len(self.aggregated_grads[0]))
                ]

                with torch.no_grad():
                    for p, g in zip(self.model.parameters(), avg_grads):
                        new_param = p + g
                        p.copy_(new_param)

                self.aggregated_grads = []
                print("Updated params")
                self.sync_done.set()
                # NOTE: Multiprocessing might be a better fit for running validation
                thr = threading.Thread(
                    name=f"Validation-{request.epoch}", target=self._run_validation
                )
                thr.start()
                return self._make_update_response()

        print(f"{request.rank} waiting for all...")
        self.sync_done.wait()

        with self.lock:
            if self.sync_done.is_set():
                self.sync_done.clear()
            print(f"Sending response to {request.rank}")
            return self._make_update_response()

    def AsyncUpdate(self, request: UpdateRequest, context: grpc.ServicerContext):
        grads = [
            deserialize_tensor(tensor_proto) / (self.world_size - 1)
            for tensor_proto in request.gradients
        ]

        with self.lock:
            self.req_cnt = (self.req_cnt + 1) % (self.world_size - 1)
            with torch.no_grad():
                for p, g in zip(self.model.parameters(), grads):
                    new_param = p + g
                    p.copy_(new_param)

            if self.req_cnt == 0:
                thr = threading.Thread(
                    name=f"Validation-{request.epoch}", target=self._run_validation
                )
                thr.start()
            return self._make_update_response()

    def _make_update_response(self) -> UpdateResponse:
        return UpdateResponse(
            parameters=[serialize_tensor(p) for p in self.model.parameters()]
        )

    def _run_validation(self):
        # HACK: Sleep here a little bit so that the lock does not get held before we send back the response to the workers
        time.sleep(3)
        with self.lock:
            print("Running validation...")
            was_training = self.model.training
            name = "Validation"

            self.model.eval()
            total_loss = 0.0
            correct = 0
            output_counter = 0
            loss_counter = 0
            y_true = []
            y_prediction = []
            batch_records = []

            with torch.no_grad():
                for i, data in enumerate(self.val_loader):
                    inputs, labels = (
                        data[0].to(self.device, non_blocking=True),
                        data[1].to(self.device, non_blocking=True),
                    )
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    total_loss += loss.item()
                    loss_counter += 1
                    prediction = outputs.argmax(dim=1, keepdim=True)
                    correct += prediction.eq(labels.view_as(prediction)).sum().item()
                    output_counter += len(labels)
                    y_prediction.extend(prediction.squeeze().tolist())
                    y_true.extend(labels.tolist())
                    batch_records.append({"batch": i + 1, "loss": loss.item()})

            total_loss /= loss_counter
            accuracy = 100.0 * correct / output_counter
            f1 = f1_score(y_true, y_prediction, average="weighted")

            print(
                f"{name} Loss: {total_loss:.4f}, {name} Accuracy: {accuracy:.2f}%, {name} F1: {f1:.4f}"
            )

            self.model.train(was_training)

    def _chunk_stream(self):
        for p in self.model.parameters():
            yield from tensor_to_chunks(p)

        for p in self.model.buffers():
            yield from tensor_to_chunks(p)
