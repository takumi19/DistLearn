from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.config.models import ModelConfig
from decentr_my_own.models.factory import build_model


class ModelFactoryTests(unittest.TestCase):
    def test_groupnorm_resnet18_forward(self) -> None:
        model = build_model(
            ModelConfig(name="resnet18", num_classes=10, normalization="groupnorm")
        )

        batch = torch.randn(2, 3, 32, 32)
        output = model(batch)

        self.assertEqual(output.shape, (2, 10))
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
