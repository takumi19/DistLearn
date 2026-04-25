from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.training.metrics import init_confusion_matrix, update_confusion_matrix


class TrainingMetricsTests(unittest.TestCase):
    def test_update_confusion_matrix_flattens_column_targets(self) -> None:
        confusion = init_confusion_matrix(3)
        logits = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )
        targets = torch.tensor([[0], [1], [2]], dtype=torch.int64)

        update_confusion_matrix(confusion, logits, targets)

        expected = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
            ],
            dtype=torch.int64,
        )
        self.assertTrue(torch.equal(confusion, expected))


if __name__ == "__main__":
    unittest.main()
