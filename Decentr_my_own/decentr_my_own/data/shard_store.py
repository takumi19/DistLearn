from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import torch

from decentr_my_own.data.manifest import DatasetManifest, ShardMeta, SplitName, load_manifest
from decentr_my_own.data.shards import ShardPayload, compute_shard_digest


class ShardStore:
    def __init__(
        self,
        manifest: DatasetManifest,
        manifest_path: str | Path,
        *,
        base_dir: str | Path | None = None,
    ):
        self.manifest = manifest
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.base_dir = self.manifest_dir if base_dir is None else Path(base_dir)
        self._shards_by_id = {shard.shard_id: shard for shard in manifest.shards}

    @classmethod
    def from_manifest_path(
        cls,
        manifest_path: str | Path,
        *,
        base_dir: str | Path | None = None,
    ) -> "ShardStore":
        path = Path(manifest_path)
        return cls(load_manifest(path), path, base_dir=base_dir)

    def list_shards(self, split: SplitName | None = None) -> list[ShardMeta]:
        shards = self.manifest.shards if split is None else self.manifest.shards_for_split(split)
        return sorted(shards, key=lambda item: item.shard_id)

    def list_local_shards(self, split: SplitName | None = None) -> list[ShardMeta]:
        return [
            shard for shard in self.list_shards(split) if self.resolve_path(shard).exists()
        ]

    def local_cache_size_bytes(self, split: SplitName | None = None) -> int:
        return sum(
            self.resolve_path(shard).stat().st_size for shard in self.list_local_shards(split)
        )

    def get_shard_meta(self, shard_id: str) -> ShardMeta:
        try:
            return self._shards_by_id[shard_id]
        except KeyError as exc:
            raise KeyError(f"Unknown shard id: {shard_id}") from exc

    def resolve_path(self, shard: str | ShardMeta) -> Path:
        shard_meta = self.get_shard_meta(shard) if isinstance(shard, str) else shard
        return self.base_dir / shard_meta.relative_path

    def has_local_shard(self, shard: str | ShardMeta) -> bool:
        return self.resolve_path(shard).exists()

    def delete_local_shard(self, shard: str | ShardMeta) -> bool:
        path = self.resolve_path(shard)
        if not path.exists():
            return False
        path.unlink()
        return True

    def trim_cache(
        self,
        *,
        max_cache_bytes: int,
        protected_shard_ids: Iterable[str] = (),
        split: SplitName | None = "train",
    ) -> list[str]:
        if max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")

        protected = set(protected_shard_ids)
        local_shards = self.list_local_shards(split)
        total_size = sum(self.resolve_path(shard).stat().st_size for shard in local_shards)
        if total_size <= max_cache_bytes:
            return []

        candidates: list[tuple[float, str, int]] = []
        for shard in local_shards:
            if shard.shard_id in protected:
                continue
            path = self.resolve_path(shard)
            stat = path.stat()
            candidates.append((stat.st_mtime, shard.shard_id, stat.st_size))

        candidates.sort()
        evicted: list[str] = []
        for _, shard_id, shard_size in candidates:
            if total_size <= max_cache_bytes:
                break
            if self.delete_local_shard(shard_id):
                total_size -= shard_size
                evicted.append(shard_id)
        return evicted

    def verify_shard(self, shard: str | ShardMeta) -> None:
        shard_meta = self.get_shard_meta(shard) if isinstance(shard, str) else shard
        verify_shard_file(self.resolve_path(shard_meta), shard_meta)

    def load_shard(self, shard: str | ShardMeta) -> ShardPayload:
        shard_meta = self.get_shard_meta(shard) if isinstance(shard, str) else shard
        self.verify_shard(shard_meta)
        payload = torch.load(self.resolve_path(shard_meta), map_location="cpu", weights_only=False)
        return ShardPayload.from_serializable(payload)


def verify_shard_file(path: str | Path, shard_meta: ShardMeta) -> None:
    shard_path = Path(path)
    blob = shard_path.read_bytes()
    if len(blob) != shard_meta.byte_size:
        raise ValueError(
            f"Shard '{shard_meta.shard_id}' size mismatch: "
            f"expected {shard_meta.byte_size}, got {len(blob)}"
        )
    payload = ShardPayload.from_serializable(
        torch.load(shard_path, map_location="cpu", weights_only=False)
    )
    digest = compute_shard_digest(payload)
    if digest != shard_meta.sha256:
        raise ValueError(
            f"Shard '{shard_meta.shard_id}' sha256 mismatch: "
            f"expected {shard_meta.sha256}, got {digest}"
        )
