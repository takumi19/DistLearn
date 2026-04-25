from __future__ import annotations

import hashlib
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchvision import datasets
from torchvision.transforms import functional as transform_functional

from decentr_my_own.config.models import TrainingConfig
from decentr_my_own.data.manifest import DatasetManifest, ShardMeta, SplitName, SplitSummary


@dataclass(frozen=True)
class ShardPayload:
    images: torch.Tensor
    labels: torch.Tensor
    sample_ids: list[int]
    meta: dict[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "labels": self.labels,
            "sample_ids": self.sample_ids,
            "meta": self.meta,
        }

    @classmethod
    def from_serializable(cls, payload: dict[str, Any]) -> "ShardPayload":
        images = payload["images"]
        labels = payload["labels"]
        sample_ids = payload["sample_ids"]
        meta = payload["meta"]

        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("Shard payload images must be a 4D tensor")
        if images.dtype != torch.uint8:
            raise ValueError("Shard payload images must use uint8 CHW tensors")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
            raise ValueError("Shard payload labels must be a 1D tensor")
        if labels.dtype != torch.int64:
            raise ValueError("Shard payload labels must use int64 tensors")
        if not isinstance(sample_ids, list) or any(not isinstance(item, int) for item in sample_ids):
            raise ValueError("Shard payload sample_ids must be a list of integers")
        if images.size(0) != labels.size(0) or labels.size(0) != len(sample_ids):
            raise ValueError("Shard payload tensors and sample_ids must have matching lengths")
        if not isinstance(meta, dict):
            raise ValueError("Shard payload meta must be a dictionary")

        return cls(images=images, labels=labels, sample_ids=sample_ids, meta=meta)


@dataclass(frozen=True)
class BuildShardsResult:
    manifest_path: Path
    dataset_name: str
    shard_count: int
    split_counts: dict[str, int]
    split_sample_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "dataset_name": self.dataset_name,
            "shard_count": self.shard_count,
            "split_counts": self.split_counts,
            "split_sample_counts": self.split_sample_counts,
        }


@dataclass(frozen=True)
class _SplitSource:
    split: SplitName
    dataset: Any
    dataset_indices: list[int]
    sample_ids: list[int]


def build_dataset_shards(
    config: TrainingConfig,
    *,
    force: bool = False,
) -> BuildShardsResult:
    if config.dataset.storage_mode != "micro_shards":
        raise ValueError("build-shards requires dataset.storage_mode=micro_shards")
    if config.dataset.manifest_path is None or config.dataset.shard_samples is None:
        raise ValueError("micro_shards mode requires dataset.manifest_path and dataset.shard_samples")

    manifest_path = Path(config.dataset.manifest_path)
    manifest_dir = manifest_path.parent
    shard_root = manifest_dir / f"{manifest_path.stem}_shards"

    if manifest_path.exists() and not force:
        raise FileExistsError(f"Manifest already exists: {manifest_path}")
    if shard_root.exists() and any(shard_root.iterdir()) and not force:
        raise FileExistsError(f"Shard directory already exists: {shard_root}")

    if force:
        if manifest_path.exists():
            manifest_path.unlink()
        if shard_root.exists():
            shutil.rmtree(shard_root)

    split_sources = _build_split_sources(config)
    image_shape = _infer_image_shape(split_sources)

    shards: list[ShardMeta] = []
    split_summaries: dict[SplitName, SplitSummary] = {}
    split_counts: dict[str, int] = {}
    split_sample_counts: dict[str, int] = {}

    for split_source in split_sources:
        shard_count = 0
        for shard_index, start_index in enumerate(
            range(0, len(split_source.dataset_indices), config.dataset.shard_samples)
        ):
            end_index = start_index + config.dataset.shard_samples
            dataset_indices = split_source.dataset_indices[start_index:end_index]
            sample_ids = split_source.sample_ids[start_index:end_index]
            payload = _build_shard_payload(
                dataset=split_source.dataset,
                dataset_indices=dataset_indices,
                sample_ids=sample_ids,
                split=split_source.split,
                dataset_name=config.dataset.name,
            )
            shard_id = f"{split_source.split}-{shard_index:05d}"
            relative_path = (
                Path(f"{manifest_path.stem}_shards")
                / split_source.split
                / f"{shard_id}.pt"
            )
            byte_size, sha256 = _write_shard_payload(payload, manifest_dir / relative_path)
            shards.append(
                ShardMeta(
                    shard_id=shard_id,
                    split=split_source.split,
                    relative_path=relative_path.as_posix(),
                    sample_count=len(sample_ids),
                    byte_size=byte_size,
                    sha256=sha256,
                )
            )
            shard_count += 1

        split_summaries[split_source.split] = SplitSummary(
            split=split_source.split,
            sample_count=len(split_source.sample_ids),
            shard_count=shard_count,
        )
        split_counts[split_source.split] = shard_count
        split_sample_counts[split_source.split] = len(split_source.sample_ids)

    manifest = DatasetManifest(
        dataset_name=config.dataset.name,
        seed=config.seed,
        shard_samples=config.dataset.shard_samples,
        num_classes=config.model.num_classes,
        image_shape=image_shape,
        splits=split_summaries,
        shards=shards,
    )
    from decentr_my_own.data.manifest import save_manifest

    save_manifest(manifest, manifest_path)
    return BuildShardsResult(
        manifest_path=manifest_path,
        dataset_name=config.dataset.name,
        shard_count=len(shards),
        split_counts=split_counts,
        split_sample_counts=split_sample_counts,
    )


def _build_split_sources(config: TrainingConfig) -> list[_SplitSource]:
    dataset_name = config.dataset.name.lower()
    if dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=config.dataset.root,
            train=True,
            download=config.dataset.download,
            transform=None,
        )
        test_dataset = datasets.CIFAR100(
            root=config.dataset.root,
            train=False,
            download=config.dataset.download,
            transform=None,
        )
        train_indices, val_indices = _split_indices(
            total_size=len(train_dataset),
            val_split=config.dataset.val_split,
            seed=config.seed,
        )
        return [
            _SplitSource(
                split="train",
                dataset=train_dataset,
                dataset_indices=train_indices,
                sample_ids=train_indices.copy(),
            ),
            _SplitSource(
                split="val",
                dataset=train_dataset,
                dataset_indices=val_indices,
                sample_ids=val_indices.copy(),
            ),
            _SplitSource(
                split="test",
                dataset=test_dataset,
                dataset_indices=list(range(len(test_dataset))),
                sample_ids=list(range(len(test_dataset))),
            ),
        ]

    if dataset_name == "fakedata":
        image_size = (3, config.dataset.image_size, config.dataset.image_size)
        train_dataset = datasets.FakeData(
            size=config.dataset.fake_train_size,
            image_size=image_size,
            num_classes=config.model.num_classes,
            transform=None,
        )
        val_dataset = datasets.FakeData(
            size=config.dataset.fake_val_size,
            image_size=image_size,
            num_classes=config.model.num_classes,
            transform=None,
        )
        test_dataset = datasets.FakeData(
            size=config.dataset.fake_test_size,
            image_size=image_size,
            num_classes=config.model.num_classes,
            transform=None,
        )
        return [
            _SplitSource(
                split="train",
                dataset=train_dataset,
                dataset_indices=list(range(len(train_dataset))),
                sample_ids=list(range(len(train_dataset))),
            ),
            _SplitSource(
                split="val",
                dataset=val_dataset,
                dataset_indices=list(range(len(val_dataset))),
                sample_ids=list(range(len(val_dataset))),
            ),
            _SplitSource(
                split="test",
                dataset=test_dataset,
                dataset_indices=list(range(len(test_dataset))),
                sample_ids=list(range(len(test_dataset))),
            ),
        ]

    raise ValueError(f"Unsupported dataset '{config.dataset.name}'. Available: CIFAR100, FakeData")


def _split_indices(total_size: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]
    return train_indices, val_indices


def _infer_image_shape(split_sources: list[_SplitSource]) -> tuple[int, int, int]:
    for split_source in split_sources:
        if split_source.dataset_indices:
            image, _ = split_source.dataset[split_source.dataset_indices[0]]
            tensor = _to_uint8_chw(image)
            return tuple(int(item) for item in tensor.shape)
    raise ValueError("Cannot infer image shape from empty datasets")


def _build_shard_payload(
    *,
    dataset: Any,
    dataset_indices: list[int],
    sample_ids: list[int],
    split: SplitName,
    dataset_name: str,
) -> ShardPayload:
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for dataset_index in dataset_indices:
        image, label = dataset[dataset_index]
        images.append(_to_uint8_chw(image))
        labels.append(int(label))

    image_tensor = torch.stack(images, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.int64)
    return ShardPayload(
        images=image_tensor,
        labels=label_tensor,
        sample_ids=sample_ids,
        meta={
            "format_version": 1,
            "split": split,
            "dataset_name": dataset_name,
            "sample_count": len(sample_ids),
        },
    )


def _to_uint8_chw(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError("Expected image tensors to be 3-dimensional")
        if tensor.dtype != torch.uint8:
            if tensor.is_floating_point():
                scale = 255.0 if tensor.max().item() <= 1.0 else 1.0
                tensor = torch.round(tensor * scale).clamp(0, 255).to(torch.uint8)
            else:
                tensor = tensor.clamp(0, 255).to(torch.uint8)
        if tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1).contiguous()
        return tensor
    return transform_functional.pil_to_tensor(image)


def _write_shard_payload(payload: ShardPayload, path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(
        payload.to_serializable(),
        buffer,
        pickle_protocol=4,
        _use_new_zipfile_serialization=False,
    )
    blob = buffer.getvalue()
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_bytes(blob)
    temp_path.replace(path)
    return len(blob), compute_shard_digest(payload)


def compute_shard_digest(payload: ShardPayload) -> str:
    hasher = hashlib.sha256()
    hasher.update(_tensor_digest_prefix("images", payload.images))
    hasher.update(payload.images.contiguous().numpy().tobytes())
    hasher.update(_tensor_digest_prefix("labels", payload.labels))
    hasher.update(payload.labels.contiguous().numpy().tobytes())
    hasher.update(
        json.dumps(payload.sample_ids, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    hasher.update(
        json.dumps(payload.meta, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    )
    return hasher.hexdigest()


def _tensor_digest_prefix(name: str, tensor: torch.Tensor) -> bytes:
    descriptor = {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    return json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
