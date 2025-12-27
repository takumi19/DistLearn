import argparse
import grpc
from concurrent import futures
from proto.ps_pb2_grpc import add_ParameterServerServicer_to_server, ParameterServerStub
from server import ParameterServerServicer
from worker import worker
from datetime import datetime

import torch
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torchvision import transforms, datasets, models


def main():
    start_time = str(datetime.now()).split(".", 1)[0].replace(" ", "T")
    args = parse_args()
    train_loader, val_loader, _, train_sampler = load_datasets(
        args.data_dir, args.batch_size, args.rank, args.world_size
    )
    model = models.resnet18(num_classes=100)

    if args.rank == 0:
        srv = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_ParameterServerServicer_to_server(
            ParameterServerServicer(model, args.world_size, val_loader),
            srv,
        )
        srv.add_insecure_port(f"[::]:{args.port}")
        srv.start()
        srv.wait_for_termination()
    else:
        channel = grpc.insecure_channel(f"localhost:{args.port}")
        ps = ParameterServerStub(channel)
        worker(
            model,
            train_loader,
            train_sampler,
            ps,
            args.rank,
            args.num_epochs,
            start_time,
        )


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
        "--master_port", type=int, default=50051, help="The master port."
    )
    parser.add_argument(
        "--sync", type=int, default=1, help="Whether to use sync or async training."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="weight decay (L2 penalty)"
    )
    return parser.parse_args()


def load_datasets(
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


if __name__ == "__main__":
    main()
