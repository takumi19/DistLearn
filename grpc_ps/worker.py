import os
from itertools import chain
from typing import Iterable

import pandas as pd
import torch
from helpers import (
    chunks_to_tensor,
    tensor_to_chunks,
)
from proto.ps_pb2 import TensorChunk
from proto.ps_pb2_grpc import ParameterServerStub
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler


def worker(
    model: torch.nn.Module,
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    ps: ParameterServerStub,
    rank: int,
    n_epochs: int,
    start_time: str,
    criterion: torch.nn.Module,
    sync: bool = True,
    max_lr=1e-2,
    streaming: bool = False,
):
    epoch_metrics = []
    batch_records = []
    device = "cpu"

    optimizer = optim.SGD(
        model.parameters(),
        lr=max_lr,
        momentum=0.9,
        weight_decay=1e-5,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, n_epochs * len(train_loader)
    )

    for epoch in range(n_epochs):
        print(f"Epoch {epoch} started")
        model.train()
        train_sampler.set_epoch(epoch)

        correct = 0
        total = 0
        epoch_loss = 0.0
        initial_params = [param.clone().detach() for param in model.parameters()]
        initial_buffers = [buffer.clone().detach() for buffer in model.buffers()]

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            predicted = output.detach().argmax(dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            epoch_loss += loss.item() * target.size(0)

            batch_records.append(
                {
                    "rank": rank,
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "loss": loss.item(),
                }
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if "TEST" in os.environ and os.environ["TEST"] == "1":
                break

        scheduler.step()

        print(f"Sending the updates to the server for {epoch=}")

        if streaming:
            metadata = (("rank", str(rank)), ("epoch", str(epoch)))
            resp_iter: Iterable[TensorChunk] = ps.StreamingSyncUpdate(
                tensor_stream(initial_params, initial_buffers, model),
                metadata=metadata,
            )

            params_and_bufs = chain(model.parameters(), model.buffers())
            chunks: list[TensorChunk] = []
            for chunk in resp_iter:
                chunks.append(chunk)
                if chunk.is_last:
                    tensor = chunks_to_tensor(chunks)
                    with torch.no_grad():
                        next(params_and_bufs).copy_(tensor)
                    chunks.clear()
        else:
            raise NotImplementedError()

        torch.save(
            model.state_dict(),
            f"./model_weights/{start_time}/worker-{rank}_epoch-{epoch + 1}.pth",
        )

        epoch_accuracy = correct / total if total else 0.0
        avg_epoch_loss = epoch_loss / total if total else 0.0
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "accuracy": epoch_accuracy,
                "train_loss_epoch": avg_epoch_loss,
            }
        )

    # After training
    metrics_df = pd.DataFrame(epoch_metrics)
    metrics_df.to_csv(f"logs/{start_time}/metrics_worker_{rank}.csv", index=False)
    pd.DataFrame(batch_records).to_csv(
        f"logs/{start_time}/batches_worker_{rank}.csv", index=False
    )
    # NOTE: Maybe add validation
    print("Done")


def tensor_stream(
    initial_params: list[torch.Tensor],
    initial_buffers: list[torch.Tensor],
    model: torch.nn.Module,
) -> Iterable[TensorChunk]:
    for new_param, initial_param in zip(model.parameters(), initial_params):
        yield from tensor_to_chunks(new_param - initial_param)

    for new_buffer, initial_buffer in zip(model.buffers(), initial_buffers):
        yield from tensor_to_chunks(new_buffer - initial_buffer)
