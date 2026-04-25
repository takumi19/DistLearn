from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SplitName = Literal["train", "val", "test"]


class SplitSummary(BaseModel):
    split: SplitName
    sample_count: int = Field(ge=0)
    shard_count: int = Field(ge=0)


class ShardMeta(BaseModel):
    shard_id: str = Field(min_length=1)
    split: SplitName
    relative_path: str = Field(min_length=1)
    sample_count: int = Field(ge=1)
    byte_size: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("Shard paths must be relative to the manifest directory")
        if ".." in path.parts:
            raise ValueError("Shard paths must not escape the manifest directory")
        return value


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: int = Field(default=1, ge=1)
    dataset_name: str = Field(min_length=1)
    storage_mode: Literal["micro_shards"] = "micro_shards"
    seed: int
    shard_samples: int = Field(ge=1)
    num_classes: int = Field(ge=2)
    image_shape: tuple[int, int, int]
    splits: dict[SplitName, SplitSummary]
    shards: list[ShardMeta] = Field(default_factory=list)

    @field_validator("image_shape")
    @classmethod
    def validate_image_shape(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(value) != 3 or any(item <= 0 for item in value):
            raise ValueError("image_shape must contain exactly three positive dimensions")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> "DatasetManifest":
        shard_ids = [shard.shard_id for shard in self.shards]
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("Shard ids must be unique within one manifest")

        relative_paths = [shard.relative_path for shard in self.shards]
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError("Shard relative paths must be unique within one manifest")

        shard_counts = {split_name: 0 for split_name in self.splits}
        sample_counts = {split_name: 0 for split_name in self.splits}
        for split_name, summary in self.splits.items():
            if summary.split != split_name:
                raise ValueError(f"Split summary key '{split_name}' must match summary.split")

        for shard in self.shards:
            if shard.split not in self.splits:
                raise ValueError(f"Shard '{shard.shard_id}' references unknown split '{shard.split}'")
            shard_counts[shard.split] += 1
            sample_counts[shard.split] += shard.sample_count

        for split_name, summary in self.splits.items():
            if summary.shard_count != shard_counts[split_name]:
                raise ValueError(
                    f"Split '{split_name}' shard_count={summary.shard_count} "
                    f"does not match manifest shards={shard_counts[split_name]}"
                )
            if summary.sample_count != sample_counts[split_name]:
                raise ValueError(
                    f"Split '{split_name}' sample_count={summary.sample_count} "
                    f"does not match manifest samples={sample_counts[split_name]}"
                )

        return self

    def shards_for_split(self, split: SplitName) -> list[ShardMeta]:
        return [shard for shard in self.shards if shard.split == split]


def load_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DatasetManifest.model_validate(payload)


def save_manifest(manifest: DatasetManifest, path: str | Path) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path
