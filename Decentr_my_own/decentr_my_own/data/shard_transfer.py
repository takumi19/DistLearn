from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from decentr_my_own.data.manifest import ShardMeta
from decentr_my_own.data.shard_store import verify_shard_file


@dataclass(frozen=True)
class PulledShardResult:
    shard_id: str
    path: Path
    byte_size: int

    def to_dict(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "path": str(self.path),
            "byte_size": self.byte_size,
        }


def iter_shard_file_chunks(
    path: str | Path,
    *,
    shard_id: str,
    chunk_size: int,
) -> Iterable[tuple[int, bytes, bool]]:
    shard_path = Path(path)
    offset = 0
    with shard_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            next_offset = offset
            offset += len(chunk)
            yield next_offset, chunk, False
    yield offset, b"", True


def write_pulled_shard(
    *,
    shard_id: str,
    destination_path: str | Path,
    chunks: Iterable[bytes],
    expected_meta: ShardMeta | None = None,
) -> PulledShardResult:
    path = Path(destination_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    num_bytes = 0
    with temp_path.open("wb") as handle:
        for chunk in chunks:
            handle.write(chunk)
            num_bytes += len(chunk)
    temp_path.replace(path)

    if expected_meta is not None:
        verify_shard_file(path, expected_meta)

    return PulledShardResult(
        shard_id=shard_id,
        path=path,
        byte_size=num_bytes,
    )
