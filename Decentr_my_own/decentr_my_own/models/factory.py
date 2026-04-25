from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d

from decentr_my_own.config.models import ModelConfig


MODEL_FACTORIES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}


def build_model(config: ModelConfig) -> nn.Module:
    try:
        factory = MODEL_FACTORIES[config.name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_FACTORIES))
        raise ValueError(f"Unsupported model '{config.name}'. Available: {available}") from exc

    model = factory(weights=None, num_classes=config.num_classes)
    if hasattr(model, "conv1"):
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
    if hasattr(model, "maxpool"):
        model.maxpool = nn.Identity()

    if config.normalization == "groupnorm":
        model = _replace_batch_norm(model, _build_group_norm)
    elif config.normalization == "frozen_batchnorm":
        model = _replace_batch_norm(model, _build_frozen_batch_norm)

    return model


def _replace_batch_norm(module: nn.Module, factory) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, factory(child.num_features))
        else:
            _replace_batch_norm(child, factory)
    return module


def _build_group_norm(num_features: int) -> nn.GroupNorm:
    groups = min(32, num_features)
    while num_features % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_features)


def _build_frozen_batch_norm(num_features: int) -> FrozenBatchNorm2d:
    return FrozenBatchNorm2d(num_features)
