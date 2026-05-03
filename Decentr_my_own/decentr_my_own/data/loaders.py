from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import datasets, transforms

from decentr_my_own.config.models import ResolvedConfig, TrainingConfig
from decentr_my_own.data.partitioning import PartitionPlanner, PartitionedIndexSampler
from decentr_my_own.data.shard_dataset import (
    FixedShardAssignmentSampler,
    ShardDataset,
    ShardWindowLoader,
    StaticShardAssignmentPlanner,
    StaticShardAssignmentSampler,
)
from decentr_my_own.data.shard_store import ShardStore

CIFAR_DATASETS = {
    "cifar10": {
        "factory": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "train_size": 50_000,
    },
    "cifar100": {
        "factory": datasets.CIFAR100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "train_size": 50_000,
    },
}


class EpochSampler(Protocol):
    def set_epoch(self, epoch: int) -> None: ...

    def __len__(self) -> int: ...


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_sampler: EpochSampler | None = None
    train_sample_count: int = 0


def build_local_dataloaders(
    resolved: ResolvedConfig,
    pin_memory: bool = False,
    *,
    train_shard_ids: list[str] | None = None,
) -> DataLoaders:
    config = resolved.training
    if config.dataset.storage_mode == "micro_shards":
        return _build_micro_shard_loaders(
            resolved,
            pin_memory,
            train_shard_ids=train_shard_ids,
        )
    dataset_name = config.dataset.name.lower()
    if dataset_name in CIFAR_DATASETS:
        return _build_cifar_loaders(resolved, pin_memory)
    if dataset_name == "fakedata":
        return _build_fake_data_loaders(resolved, pin_memory)
    raise ValueError(
        f"Unsupported dataset '{config.dataset.name}'. Available: CIFAR10, CIFAR100, FakeData"
    )


def _build_cifar_loaders(
    resolved: ResolvedConfig, pin_memory: bool
) -> DataLoaders:
    config = resolved.training
    dataset_spec = CIFAR_DATASETS[config.dataset.name.lower()]
    dataset_factory = dataset_spec["factory"]
    transform_train, transform_eval = _build_cifar_transforms(config, storage_mode="replicated")

    train_dataset_aug = dataset_factory(
        root=config.dataset.root,
        train=True,
        download=config.dataset.download,
        transform=transform_train,
    )
    train_dataset_eval = dataset_factory(
        root=config.dataset.root,
        train=True,
        download=False,
        transform=transform_eval,
    )
    test_dataset = dataset_factory(
        root=config.dataset.root,
        train=False,
        download=config.dataset.download,
        transform=transform_eval,
    )

    train_indices, val_indices = _split_indices(
        len(train_dataset_aug), config.dataset.val_split, config.seed
    )

    train_subset = Subset(train_dataset_aug, train_indices)
    val_subset = Subset(train_dataset_eval, val_indices)
    sampler = _build_train_sampler(resolved, len(train_subset))

    return DataLoaders(
        train=_make_loader(
            train_subset,
            config,
            shuffle=False,
            pin_memory=pin_memory,
            sampler=sampler,
        ),
        val=_make_loader(val_subset, config, shuffle=False, pin_memory=pin_memory),
        test=_make_loader(test_dataset, config, shuffle=False, pin_memory=pin_memory),
        train_sampler=sampler,
        train_sample_count=len(sampler),
    )


def _build_fake_data_loaders(resolved: ResolvedConfig, pin_memory: bool) -> DataLoaders:
    config = resolved.training
    size = (3, config.dataset.image_size, config.dataset.image_size)
    transform = _build_fake_data_transform(storage_mode="replicated")

    train_dataset = datasets.FakeData(
        size=config.dataset.fake_train_size,
        image_size=size,
        num_classes=config.model.num_classes,
        transform=transform,
    )
    val_dataset = datasets.FakeData(
        size=config.dataset.fake_val_size,
        image_size=size,
        num_classes=config.model.num_classes,
        transform=transform,
    )
    test_dataset = datasets.FakeData(
        size=config.dataset.fake_test_size,
        image_size=size,
        num_classes=config.model.num_classes,
        transform=transform,
    )
    sampler = _build_train_sampler(resolved, len(train_dataset))

    return DataLoaders(
        train=_make_loader(
            train_dataset,
            config,
            shuffle=False,
            pin_memory=pin_memory,
            sampler=sampler,
        ),
        val=_make_loader(val_dataset, config, shuffle=False, pin_memory=pin_memory),
        test=_make_loader(test_dataset, config, shuffle=False, pin_memory=pin_memory),
        train_sampler=sampler,
        train_sample_count=len(sampler),
    )


def _build_micro_shard_loaders(
    resolved: ResolvedConfig,
    pin_memory: bool,
    *,
    train_shard_ids: list[str] | None = None,
) -> DataLoaders:
    config = resolved.training
    if config.dataset.manifest_path is None:
        raise ValueError("dataset.manifest_path is required for storage_mode=micro_shards")

    store = ShardStore.from_manifest_path(
        config.dataset.manifest_path,
        base_dir=config.dataset.cache_dir,
    )
    if store.manifest.dataset_name.lower() != config.dataset.name.lower():
        raise ValueError(
            "Manifest dataset name does not match training config: "
            f"{store.manifest.dataset_name} != {config.dataset.name}"
        )

    if config.dataset.name.lower() in CIFAR_DATASETS:
        train_transform, eval_transform = _build_cifar_transforms(
            config,
            storage_mode="micro_shards",
        )
    elif config.dataset.name.lower() == "fakedata":
        train_transform = eval_transform = _build_fake_data_transform(
            storage_mode="micro_shards"
        )
    else:
        raise ValueError(
            f"Unsupported dataset '{config.dataset.name}'. Available: CIFAR10, CIFAR100, FakeData"
        )

    train_dataset = ShardDataset(store, split="train", transform=train_transform)
    val_dataset = ShardDataset(store, split="val", transform=eval_transform)
    test_dataset = ShardDataset(store, split="test", transform=eval_transform)
    train_sampler = _build_micro_shard_sampler(
        resolved,
        train_dataset,
        train_shard_ids=train_shard_ids,
    )

    return DataLoaders(
        train=ShardWindowLoader(
            train_dataset,
            batch_size=config.optimization.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=config.dataset.num_workers,
            pin_memory=pin_memory,
        ),
        val=ShardWindowLoader(
            val_dataset,
            batch_size=config.optimization.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            pin_memory=pin_memory,
        ),
        test=ShardWindowLoader(
            test_dataset,
            batch_size=config.optimization.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            pin_memory=pin_memory,
        ),
        train_sampler=train_sampler,
        train_sample_count=len(train_sampler),
    )


def _make_loader(
    dataset: Dataset,
    config: TrainingConfig,
    *,
    shuffle: bool,
    pin_memory: bool,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
    )


def estimate_train_sample_count(config: TrainingConfig) -> int:
    dataset_name = config.dataset.name.lower()
    if dataset_name == "fakedata":
        return config.dataset.fake_train_size
    if dataset_name in CIFAR_DATASETS:
        full_train_size = CIFAR_DATASETS[dataset_name]["train_size"]
        return full_train_size - int(full_train_size * config.dataset.val_split)
    raise ValueError(
        f"Unsupported dataset '{config.dataset.name}'. Available: CIFAR10, CIFAR100, FakeData"
    )


def build_partition_summary(
    resolved: ResolvedConfig, shuffle_token: int = 0
) -> dict:
    total_size = estimate_train_sample_count(resolved.training)
    planner = PartitionPlanner(resolved.cluster, resolved.training)
    return planner.describe_node(
        node_id=resolved.self_node_id,
        total_size=total_size,
        shuffle_token=shuffle_token,
    )


def _split_indices(total_size: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]
    return train_indices, val_indices


def _build_train_sampler(
    resolved: ResolvedConfig, total_size: int
) -> PartitionedIndexSampler:
    planner = PartitionPlanner(resolved.cluster, resolved.training)
    return PartitionedIndexSampler(
        planner=planner,
        node_id=resolved.self_node_id,
        total_size=total_size,
    )


def _build_micro_shard_sampler(
    resolved: ResolvedConfig,
    train_dataset: ShardDataset,
    *,
    train_shard_ids: list[str] | None = None,
) -> StaticShardAssignmentSampler:
    if train_shard_ids is not None:
        return FixedShardAssignmentSampler(
            dataset=train_dataset,
            training=resolved.training,
            shard_ids=train_shard_ids,
        )
    planner = StaticShardAssignmentPlanner(
        resolved.cluster,
        resolved.training,
        train_dataset.shards,
        split="train",
    )
    return StaticShardAssignmentSampler(
        planner=planner,
        dataset=train_dataset,
        node_id=resolved.self_node_id,
    )


def _build_cifar_transforms(
    config: TrainingConfig,
    *,
    storage_mode: str,
) -> tuple[transforms.Compose, transforms.Compose]:
    image_size = config.dataset.image_size
    dataset_spec = CIFAR_DATASETS[config.dataset.name.lower()]
    mean = dataset_spec["mean"]
    std = dataset_spec["std"]
    if storage_mode == "micro_shards":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean, std),
            ]
        )
        transform_eval = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean, std),
            ]
        )
        return transform_train, transform_eval

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform_train, transform_eval


def _build_fake_data_transform(*, storage_mode: str) -> transforms.Compose:
    if storage_mode == "micro_shards":
        return transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
        ]
    )
