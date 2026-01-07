import os
import threading
import time
from datetime import datetime
from itertools import chain
from typing import Iterable

import grpc
import pandas as pd
import proto.ps_pb2_grpc as ps_grpc
import torch
from helpers import (
    chunks_to_tensor,
    tensor_to_chunks,
)
from proto.ps_pb2 import (
    GetStartTimeArgs,
    GetStartTimeReply,
    TensorChunk,
)
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
        self.start_time = str(datetime.now()).split(".", 1)[0].replace(" ", "T")
        os.makedirs(f"model_weights/{self.start_time}", exist_ok=True)
        os.makedirs(f"logs/{self.start_time}", exist_ok=True)

    def SyncUpdate(
        self, request_iterator: Iterable[TensorChunk], context: grpc.ServicerContext
    ) -> Iterable[TensorChunk]:
        param_updates, rank, epoch = self._get_params(request_iterator, context)
        print(f"S{rank} sent updates for E{epoch}")

        with self.lock:
            self.aggregated_grads.append(param_updates)
            should_wait = len(self.aggregated_grads) != self.world_size - 1

        # NOTE: this block is not protected by a mutex but it will only be invoked by one worker
        # and only after the other calls are waiting on sync_done.
        if not should_wait:
            print(f"S{rank} updating params")
            avg_grads = [
                sum(g[i] for g in self.aggregated_grads) / (self.world_size - 1)
                for i in range(len(self.aggregated_grads[0]))
            ]
            print(f"S{rank} averaged params")

            with torch.no_grad():
                params_and_bufs = chain(self.model.parameters(), self.model.buffers())
                for p, g in zip(params_and_bufs, avg_grads):
                    p.copy_(p + g)
            print(f"S{rank} updated params")

            self.aggregated_grads = []

            thr = threading.Thread(
                name=f"Validation-{epoch}", target=self._run_validation, args=(epoch,)
            )
            thr.start()

            print(f"S{rank} signals done")
            self.sync_done.set()
        else:
            print(f"S{rank} waiting")
            self.sync_done.wait()

        with self.lock:
            if self.sync_done.is_set():
                print(f"S{rank} clears condvar")
                self.sync_done.clear()

        print(f"S{rank} receiving chunks")
        yield from self._chunk_stream()

    def AsyncUpdate(
        self, request_iterator: Iterable[TensorChunk], context: grpc.ServicerContext
    ) -> Iterable[TensorChunk]:
        param_updates, rank, epoch = self._get_params(request_iterator, context)
        print(f"S{rank} sent params for E{epoch}")

        with self.lock:
            self.req_cnt = (self.req_cnt + 1) % (self.world_size - 1)
            avg_updates = [
                sum(g[i] for g in param_updates) / (self.world_size - 1)
                for i in range(len(param_updates))
            ]

            with torch.no_grad():
                params_and_bufs = chain(self.model.parameters(), self.model.buffers())
                for p, g in zip(params_and_bufs, avg_updates):
                    p.copy_(p + g)

            if self.req_cnt == 0:
                thr = threading.Thread(
                    name=f"Validation-{epoch}",
                    target=self._run_validation,
                    args=(epoch,),
                )
                thr.start()

        with self.lock:
            print(f"S{rank} receiving updates for E{epoch}")
            yield from self._chunk_stream()

    def GetStartTime(
        self, request: GetStartTimeArgs, context: grpc.ServicerContext
    ) -> GetStartTimeReply:
        return GetStartTimeReply(timestamp=self.start_time)

    def _run_validation(self, epoch: int):
        # HACK: Sleep here a little bit so that the lock does not get held before we send back the response to the workers
        time.sleep(3)
        with self.lock:
            print("Running validation...")
            was_training = self.model.training
            name = f"Validation {epoch}"

            self.model.eval()
            total_loss = 0.0
            correct = 0
            output_counter = 0
            loss_counter = 0
            y_true = []
            y_prediction = []

            with torch.no_grad():
                for data in self.val_loader:
                    inputs, labels = (
                        data[0].to(self.device, non_blocking=True),
                        data[1].to(self.device, non_blocking=True),
                    )
                    outputs = self.model(inputs)

                    total_loss += self.criterion(outputs, labels).item()
                    loss_counter += 1
                    prediction = outputs.argmax(dim=1, keepdim=True)
                    correct += prediction.eq(labels.view_as(prediction)).sum().item()
                    output_counter += len(labels)
                    y_prediction.extend(prediction.squeeze().tolist())
                    y_true.extend(labels.tolist())

            total_loss /= loss_counter
            acc = 100.0 * correct / output_counter
            f1 = f1_score(y_true, y_prediction, average="weighted")

            print(f"{name} Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%, F1: {f1:.4f}")

            val_metrics = []
            val_metrics.append(
                {
                    "epoch": epoch + 1,
                    "loss": total_loss,
                    "accuracy": acc / 100.0,
                    "f1": f1,
                }
            )
            val_df = pd.DataFrame(val_metrics)
            filename = f"logs/{self.start_time}/validation_metrics.csv"
            val_df.to_csv(
                filename, index=False, mode="a", header=not os.path.exists(filename)
            )

            self.model.train(was_training)

    def _get_params(
        self, request_iterator: Iterable[TensorChunk], context: grpc.ServicerContext
    ):
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

        return param_updates, rank, epoch

    def _chunk_stream(self):
        for p in self.model.parameters():
            yield from tensor_to_chunks(p)

        for p in self.model.buffers():
            yield from tensor_to_chunks(p)
