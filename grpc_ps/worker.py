import time

import pandas as pd
import torch
from proto.ps_pb2 import UpdateRequest
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
    sync: bool = True,
):
    tt0 = time.time()
    epoch_metrics = []
    batch_records = []
    val_metrics = []
    device = "cpu"

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        # weight_decay=weight_decay,
        nesterov=True,
    )

    for epoch in range(n_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        correct = 0
        total = 0
        epoch_loss = 0.0

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

        # After epoch
        req = UpdateRequest()
        if sync:
            resp = ps.SyncUpdate(req)
        else:
            resp = ps.AsyncUpdate(req)
        # TODO: Properly update model parameters
        _ = resp

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
    # TODO: Maybe add validation
