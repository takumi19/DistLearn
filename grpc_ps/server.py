import threading

import grpc
import proto.ps_pb2_grpc as ps_grpc
import torch
from helpers import deserialize_tensor, serialize_tensor
from proto.ps_pb2 import UpdateRequest, UpdateResponse
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
        self.last_epoch = 0
        self.sync_done = threading.Event()

    # TODO: Maybe making this async and using asyncio is a good idea
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
                # TODO: Run validation
                return self._make_update_response()

        print("Waiting for all...")
        self.sync_done.wait()

        with self.lock:
            if self.sync_done.is_set():
                self.sync_done.clear()
            print("Sending response")
            return self._make_update_response()

    def AsyncUpdate(self, request: UpdateRequest, context: grpc.ServicerContext):
        # TODO: Should probably divide by the number of workers here
        grads = [deserialize_tensor(tensor_proto) for tensor_proto in request.gradients]

        with self.lock:
            with torch.no_grad():
                for p, g in zip(self.model.parameters(), grads):
                    new_param = p + g
                    p.copy_(new_param)

            # TODO: if % world size - 1 -> run validation
            return self._make_update_response()

    def _make_update_response(self) -> UpdateResponse:
        # WARN: Not sure if parameters and regular tensors can be used interchangeably
        return UpdateResponse(
            parameters=[serialize_tensor(p) for p in self.model.parameters()]
        )

    def _run_validation(self):
        self.lock.acquire()
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

        self.lock.release()
