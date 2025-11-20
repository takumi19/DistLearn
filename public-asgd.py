import argparse
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
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed.rpc as rpc
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
    """
    The parameter server (PS) updates model parameters with gradients from the workers
    and sends the updated parameters back to the workers.
    """

    def __init__(self, model, num_workers, lr, sync, weight_decay, epoch_n):
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.num_workers = num_workers
        self.sync = sync
        print(f"{sync=}")
        self.grads: List[List[torch.Tensor]] = []
        # initialize model parameters
        assert model in model_dict.keys(), (
            f"model {model} is not in the model list: {list(model_dict.keys())}"
        )
        self.model = model_dict[model](num_classes=10)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )  # Изменение свертки c 7X7 на 3X3
        self.model.maxpool = nn.Identity()  # Убираем maxpooling
        self.model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 10))
        # zero gradients
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epoch_n
        )
        # self.bf = torch.futures.Future()

    def get_model(self):
        return self.model

    # Caller holds lock
    def _step_no_sync(self, grads, worker_rank):
        print(f"PS updates parameters based on gradients from worker{worker_rank}")
        for p, g in zip(self.model.parameters(), grads):
            p.grad = g
        self.optimizer.step()
        self.optimizer.zero_grad()

        fut = self.future_model

        fut.set_result(self.model)
        self.future_model = torch.futures.Future()
        return fut

    # Caller holds lock
    def _step_sync(self, grads, worker_rank, update_lr=False):
        self.grads.append(grads)
        print(
            f"PS received grads from worker{worker_rank} "
            f"({len(self.grads)}/{self.num_workers - 1})"
        )

        if self.future_model is None:
            self.future_model = torch.futures.Future()
        fut = self.future_model

        if len(self.grads) == self.num_workers - 1:
            avg_grads = [
                sum(g[i] for g in self.grads) / self.num_workers
                for i in range(len(self.grads[0]))
            ]
            for p, g in zip(self.model.parameters(), avg_grads):
                p.grad = g
            self.optimizer.step()
            self.optimizer.zero_grad()

            #######
            if update_lr:
                self.scheduler.step()
                # torch.save(
                #     self.model.state_dict(), f"./model_weights/{time.time()}-resnet-cifar-ps.pth"
                # )
                # print("parameter server saves model")
            #######

            fut.set_result(self.model)
            self.future_model = None

            self.grads.clear()
            self.grads = []
        return fut

    @staticmethod
    @rpc.functions.async_execution
    def update_lr(ps_rref):
        self = ps_rref.local_value()
        if not self.sync:
            return
        with self.lock:
            self.scheduler.step()

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads, worker_rank, update_lr=False):
        self = ps_rref.local_value()
        with self.lock:
            if not self.sync:
                fut = self._step_no_sync(grads, worker_rank)
            else:
                fut = self._step_sync(grads, worker_rank, update_lr)

        return fut


def stats(model, device, loader, criterion, name):
    model.eval()
    total_loss = 0.0
    correct = 0
    output_counter = 0
    loss_counter = 0
    y_true = []
    y_prediction = []
    with torch.no_grad():
        print(f"{len(loader)=}")
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
            print(f"{i=}")
    total_loss /= loss_counter
    accuracy = 100.0 * correct / output_counter
    f1 = f1_score(y_true, y_prediction, average="weighted")

    print(f"{name} Loss: {total_loss}, {name} Accuracy: {accuracy}, {name} F1: {f1}")
    return total_loss, accuracy, f1


def run_worker(ps_rref, rank, world_size, data_dir, batch_size, num_epochs):
    """
    A worker pulls model parameters from the PS, computes gradients on a mini-batch
    from its data partition, and pushes the gradients to the PS.
    """

    # prepare dataset
    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    # )

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    sampler = DistributedSampler(
        dataset=train_dataset, num_replicas=world_size - 1, rank=rank - 1
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False
    )
    eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # set device
    # device_id = rank - 1
    # device_id = 0
    # device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    criterion = nn.CrossEntropyLoss()

    # get initial model from the PS
    m = ps_rref.rpc_sync().get_model().to(device)

    print(f"worker{rank} starts training")
    tt0 = time.time()

    epoch_metrics = []

    for i in range(num_epochs):
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
            loss.backward()

            print(
                "worker{:d} | Epoch:{:3d} | Batch: {:3d} | Loss: {:6.2f}".format(
                    rank, (i + 1), (batch_idx + 1), loss.item()
                )
            )

            # only update lr from the first worker
            update_lr = (batch_idx == len(train_loader) - 1) and rank == 1
            # send gradients to the PS and fetch updated model parameters
            m = rpc.rpc_sync(
                to=ps_rref.owner(),
                func=ParameterServer.update_and_fetch_model,
                args=(ps_rref, [p.grad for p in m.cpu().parameters()], rank, update_lr),
            ).to(device)

            # Imitate stragglers or network issues
            # if rank == 1 and world_size > 2:
            #     time.sleep(random.uniform(0.1, 1.0))
        if rank == 1:
            # ps_rref.rpc_sync().update_lr()
            torch.save(
                m.state_dict(), f"./model_weights/{time.time()}-resnet-cifar-ps.pth"
            )
        epoch_accuracy = correct / total if total else 0.0
        avg_epoch_loss = epoch_loss / total if total else 0.0
        epoch_metrics.append(
            {"epoch": i + 1, "accuracy": epoch_accuracy, "loss": avg_epoch_loss}
        )

    tt1 = time.time()

    print("Time: {:.2f} seconds".format((tt1 - tt0)))
    metrics_df = pd.DataFrame(epoch_metrics)
    metrics_df.to_csv(f"logs/worker{rank}_metrics.csv", index=False)
    print(f"worker{rank} metrics by epoch:\n{metrics_df}")

    m = ps_rref.rpc_sync().get_model().to(device)
    stats(m, device, eval_loader, criterion, "test")
    # print(f"On testing data: {total_loss=}, {accuracy=}, {f1=}")


def main():
    parser = argparse.ArgumentParser(
        description="Train models on Imagenette under ASGD"
    )
    parser.add_argument("--model", type=str, default="resnet18", help="The job's name.")
    parser.add_argument(
        "--rank", type=int, default=1, help="Global rank of this process."
    )
    parser.add_argument(
        "--world_size", type=int, default=2, help="Total number of workers."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./imagenette2/val",
        help="The location of dataset.",
    )
    parser.add_argument(
        "--master_addr", type=str, default="localhost", help="Address of master."
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="29600",
        help="Port that master is listening on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size of each worker during training.",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs.")
    parser.add_argument(
        "--sync", dest="sync", action="store_true", help="Sync gradients or no."
    )

    args = parser.parse_args()
    print(f"{args.sync=}")

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16, rpc_timeout=999999999
    )

    if args.rank == 0:
        """
        initialize PS and run workers
        """
        print(f"PS{args.rank} initializing")
        rpc.init_rpc(
            f"PS{args.rank}",
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=options,
        )
        print(f"PS{args.rank} initialized")

        weight_decay = 5e-3
        ps_rref = rpc.RRef(
            ParameterServer(
                args.model,
                args.world_size,
                args.lr,
                args.sync,
                weight_decay,
                args.num_epochs,
            )
        )

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
                    ),
                )
            )

        torch.futures.wait_all(futs)
        print("Finish training")

    else:
        """
        initialize workers
        """
        print(f"worker{args.rank} initializing")
        rpc.init_rpc(
            f"worker{args.rank}",
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=options,
        )
        print(f"worker{args.rank} initialized")

    rpc.shutdown()


if __name__ == "__main__":
    main()
