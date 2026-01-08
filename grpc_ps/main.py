import argparse
from concurrent import futures

import grpc
import torch
from proto.ps_pb2_grpc import (
    ParameterServerStub,
    add_ParameterServerServicer_to_server,
)
from server import ParameterServerServicer
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torchvision import datasets, models, transforms
from worker import worker

# XXX: Maybe increasing the max message length is not a great idea, we can stream the tensors one by one
MAX_MESSAGE_LENGTH = -1


def main():
    args = parse_args()
    train_loader, val_loader, _, train_sampler = load_datasets(
        args.data_dir, args.batch_size, args.rank, args.world_size
    )
    model = models.resnet18(num_classes=100)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 100)
    criterion = torch.nn.CrossEntropyLoss()

    server_addr = f"{args.master_addr}:{args.master_port}"
    if args.rank == 0:
        srv = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_ParameterServerServicer_to_server(
            ParameterServerServicer(model, args.world_size, val_loader, criterion),
            srv,
        )
        print(f"Starting the server on {server_addr}")
        srv.add_insecure_port(server_addr)
        srv.start()
        srv.wait_for_termination()
    else:
        with grpc.insecure_channel(server_addr) as channel:
            print("Starting the client")
            ps = ParameterServerStub(channel)
            worker(
                model,
                train_loader,
                train_sampler,
                ps,
                args.rank,
                args.num_epochs,
                criterion,
                args.sync,
                args.lr,
                args.weight_decay
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
        "--batch_size", type=int, default=64, help="The number of images per batch."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=180, help="The number of epochs for training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="The learning rate for SGD with momentum.",
    )
    parser.add_argument(
        "--master_addr", type=str, default="0.0.0.0", help="The hostname of master."
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
) -> tuple[DataLoader | None, DataLoader, DataLoader | None, DistributedSampler | None]:
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(2.8),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),  # Convert images to Tensor ( Channel X Height X Width)
            transforms.Normalize((0.4911, 0.4820, 0.4467), (0.2022, 0.1993, 0.2009)),
        ]
    )

    train = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )
    test = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )
    val_ratio = 0.2
    train_data, val_data = torch.utils.data.random_split(
        train,
        [
            int((1 - val_ratio) * len(train)),
            int((val_ratio) * len(train)),
        ],
    )
    # train_dataset, val_dataset = random_split(
    #     full_train_dataset,
    #     [train_size, val_size],
    #     generator=torch.Generator(),
    # )
    # test_dataset = datasets.CIFAR100(
    #     root=data_dir, train=False, download=False, transform=transform_val
    # )

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    if rank == 0:
        return None, val_loader, None, None

    sampler = DistributedSampler(
        dataset=train_data, num_replicas=world_size - 1, rank=rank - 1
    )
    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=sampler, shuffle=False
    )
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, sampler


if __name__ == "__main__":
    main()
