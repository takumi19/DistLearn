import argparse
from datetime import datetime
import pandas as pd
import random
from typing import List
import os
import threading
import time
from sklearn.metrics import f1_score

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, random_split
import torch.distributed.rpc as rpc

torch.manual_seed(42)
random.seed(42)

from torchvision import transforms, datasets, models

model_dict = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "vgg16": models.vgg16,
    "alexnet": models.alexnet,
    "googlenet": models.googlenet,
    "inception": models.inception_v3,
    "densenet121": models.densenet121,
    "mobilenet": models.mobilenet_v2,
}


class ParameterServer(object):
    def __init__(self, model, num_workers):
        self.num_workers = num_workers
        self.starttime = str(datetime.now()).split(".", 1)[0].replace(" ", "T")

        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.epoch_barrier_count = 0
        self.epoch_barrier_future = None
        self.params: List[List[torch.Tensor]] = []

        assert model in model_dict.keys(), (
            f"model {model} is not in the model list: {list(model_dict.keys())}"
        )

        self.model = model_dict[model](num_classes=100)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 100))

    def get_start_time(self) -> str:
        return self.starttime

    def get_model(self) -> nn.Module:
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def barrier(ps_rref):
        """
        Барьер по эпохам:

        - каждый воркер в конце эпохи вызывает эту функцию;
        - PS ждёт, пока отметятся все (num_workers - 1) воркеров;
        - потом выпускает их в следующую эпоху.
        """
        self = ps_rref.local_value()
        with self.lock:
            if self.epoch_barrier_future is None:
                self.epoch_barrier_future = torch.futures.Future()
            fut = self.epoch_barrier_future
            self.epoch_barrier_count += 1
            if self.epoch_barrier_count == self.num_workers - 1:
                fut.set_result(True)
                self.epoch_barrier_future = None
                self.epoch_barrier_count = 0
        return fut

    # Caller holds lock
    def _aggregate_params(self, params, worker_rank):
        """
        Принимает параметры (веса) от воркера, копит их,
        усредняет по всем воркерам, обновляет глобальную модель на PS
        и возвращает воркерам список усреднённых параметров.
        """
        self.params.append(params)
        print(
            f"PS received params from worker{worker_rank} "
            f"({len(self.params)}/{self.num_workers - 1})"
        )

        if self.future_model is None:
            self.future_model = torch.futures.Future()
        fut = self.future_model

        if len(self.params) == self.num_workers - 1:
            # params: список длины (num_workers-1),
            # каждый элемент — список параметров модели.
            avg_params = [
                sum(g[i] for g in self.params) / (self.num_workers - 1)
                for i in range(len(self.params[0]))
            ]

            with torch.no_grad():
                for p, avg in zip(self.model.parameters(), avg_params):
                    p.copy_(avg.to(p.device))

            fut.set_result(avg_params)
            self.future_model = None

            self.params.clear()
            self.params = []
        return fut

    @staticmethod
    @rpc.functions.async_execution
    def update_lr(ps_rref):
        # lr-шедулер теперь на воркерах
        return

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, params, worker_rank):
        """
        RPC-функция для воркеров.

        - params — список тензоров параметров (весов), присланных воркером;
        - возвращает Future со списком усреднённых параметров avg_params.
        """
        self = ps_rref.local_value()
        with self.lock:
            fut = self._aggregate_params(params, worker_rank)
        return fut


def stats(
    model: nn.Module, device: str, loader: DataLoader, criterion: nn.Module, name: str
) -> tuple[float, float, float, pd.DataFrame]:
    """
    Валидация (или тест):

    - временно переключает модель в eval;
    - считает loss, accuracy, F1;
    - потом возвращает модель в исходный режим (train/eval).
    """
    was_training = model.training

    model.eval()
    total_loss = 0.0
    correct = 0
    output_counter = 0
    loss_counter = 0
    y_true = []
    y_prediction = []
    batch_records = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = (
                data[0].to(device, non_blocking=True),
                data[1].to(device, non_blocking=True),
            )
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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

    if was_training:
        model.train()

    return total_loss, accuracy, f1, pd.DataFrame(batch_records)


def run_worker(
    ps_rref: rpc.RRef[ParameterServer],
    rank: int,
    world_size: int,
    data_dir: str,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
):
    train_loader, val_loader, test_loader, train_sampler = get_datasets(
        data_dir, batch_size, rank, world_size
    )
    # ps_rref.rpc_sync().barrier()
    rpc.rpc_sync(to=ps_rref.owner(), func=ParameterServer.barrier, args=(ps_rref,))

    device = "cpu"
    criterion = nn.CrossEntropyLoss()
    m: nn.Module = ps_rref.rpc_sync().get_model().to(device)

    optimizer = optim.SGD(
        m.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"Worker-{rank} starts training")

    tt0 = time.time()
    epoch_metrics = []
    batch_records = []
    val_metrics = []
    timestamp = ps_rref.rpc_sync().get_start_time()

    for epoch in range(num_epochs):
        m.train()
        train_sampler.set_epoch(epoch)

        correct = 0
        total = 0
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = m(data)
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

            print(
                "worker{:d} | Epoch:{:3d} | Batch: {:3d} | Loss: {:6.2f}".format(
                    rank, (epoch + 1), (batch_idx + 1), loss.item()
                )
            )

        # for p in m.parameters():
        #     if p.grad is not None:
        #         p.grad = p.grad / len(train_loader)

        params_to_send = [p.detach().cpu() for p in m.parameters()]
        # avg_params = ps_rref.rpc_sync().update_and_fetch_model(
        #     ps_rref, params_to_send, rank
        # )
        avg_params = rpc.rpc_sync(
            to=ps_rref.owner(),
            func=ParameterServer.update_and_fetch_model,
            args=(ps_rref, params_to_send, rank),
        )

        with torch.no_grad():
            for p, avg in zip(m.parameters(), avg_params):
                p.copy_(avg.to(p.device))

        scheduler.step()

        torch.save(
            m.state_dict(),
            f"./model_weights/{timestamp}/worker-{rank}_epoch-{epoch + 1}.pth",
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

        if rank == 1:
            val_loss, val_acc, val_f1, val_batches = stats(
                m, device, val_loader, criterion, f"Validation (epoch {epoch + 1})"
            )
            val_metrics.append(
                {
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc / 100.0,
                    "val_f1": val_f1,
                }
            )
            val_batches = val_batches.copy()
            val_batches["epoch"] = epoch + 1
            val_batches.to_csv(
                f"logs/{timestamp}/batches_val_epoch{epoch + 1}.csv", index=False
            )

        # ps_rref.rpc_sync().barrier()
        rpc.rpc_sync(to=ps_rref.owner(), func=ParameterServer.barrier, args=(ps_rref,))

    tt1 = time.time()
    print("Time: {:.2f} seconds".format((tt1 - tt0)))

    metrics_df = pd.DataFrame(epoch_metrics)
    metrics_df.to_csv(f"logs/{timestamp}/metrics_worker_{rank}.csv", index=False)
    pd.DataFrame(batch_records).to_csv(
        f"logs/{timestamp}/batches_worker_{rank}.csv", index=False
    )

    if rank == 1 and val_metrics:
        val_df = pd.DataFrame(val_metrics)
        val_df.to_csv(f"logs/{timestamp}/validation_metrics.csv", index=False)

        print("=== FINAL TEST EVALUATION ===")
        test_loss, test_acc, test_f1, _ = stats(
            m, device, test_loader, criterion, "FINAL TEST"
        )
        final_test_df = pd.DataFrame(
            [
                {
                    "test_loss": test_loss,
                    "test_accuracy": test_acc / 100.0,
                    "test_f1": test_f1,
                    "num_epochs": num_epochs,
                }
            ]
        )
        final_test_df.to_csv(f"logs/{timestamp}/final_test_metrics.csv", index=False)
        print(f"Final Test Accuracy: {test_acc:.2f}%, F1: {test_f1:.4f}")

    print(f"Worker-{rank} metrics by epoch:\n{metrics_df}")


def main():
    args = parse_args()
    _ = datasets.CIFAR100(root=args.data_dir, download=True)

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=999999999,
    )

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    if args.rank == 0:
        rpc.init_rpc(
            "ps",
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=options,
        )
        ps_rref = rpc.RRef(
            ParameterServer(
                args.model,
                args.world_size,
            )
        )
        print("PS initialized")

        timestamp = ps_rref.rpc_sync().get_start_time()
        os.makedirs(f"model_weights/{timestamp}")
        os.makedirs(f"logs/{timestamp}")

        futs = []
        for r in range(1, args.world_size):
            worker = f"worker{r}"
            futs.append(
                rpc.rpc_async(
                    to=worker,
                    func=run_worker,
                    args=(
                        ps_rref,
                        r,
                        args.world_size,
                        args.data_dir,
                        args.batch_size,
                        args.num_epochs,
                        args.lr,
                        args.weight_decay,
                    ),
                )
            )

        torch.futures.wait_all(futs)
        print("Finished training")

        torch.save(
            ps_rref.local_value().model.state_dict(),
            f"./model_weights/{timestamp}/global_model.pth",
        )
    else:
        print(f"Worker{args.rank} initializing")
        rpc.init_rpc(
            f"worker{args.rank}",
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=options,
        )
        print(f"Worker{args.rank} initialized")

    rpc.shutdown()


def get_datasets(
    data_dir: str, batch_size: int, rank: int, world_size: int
) -> tuple[DataLoader, DataLoader, DataLoader, DistributedSampler]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    full_train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=transform_train
    )
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator(),
    )
    val_dataset.dataset.transform = transform_val
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=False, transform=transform_val
    )

    sampler = DistributedSampler(
        dataset=train_dataset, num_replicas=world_size - 1, rank=rank - 1
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", help="Model name.")
    parser.add_argument(
        "--rank", type=int, default=1, help="Global rank of this process."
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="Total number of processes (PS + workers).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="The location of dataset.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The number of images per batch."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=90, help="The number of epochs for training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="The learning rate for SGD with momentum.",
    )
    parser.add_argument(
        "--master_addr", type=str, default="localhost", help="The hostname of master."
    )
    parser.add_argument(
        "--master_port", type=int, default=29500, help="The master port."
    )
    parser.add_argument(
        "--sync", type=int, default=1, help="Whether to use sync or async training."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="weight decay (L2 penalty)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
