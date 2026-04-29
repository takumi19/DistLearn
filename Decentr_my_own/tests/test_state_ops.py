from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decentr_my_own.training.state_ops import check_state_finite


class StateOpsTests(unittest.TestCase):
    def test_finite_state_passes(self) -> None:
        report = check_state_finite(
            {
                "weight": torch.tensor([1.0, 2.0]),
                "counter": torch.tensor([1], dtype=torch.int64),
            }
        )

        self.assertTrue(report.ok)
        self.assertEqual(report.bad_tensor_names, ())
        self.assertEqual(report.nan_count, 0)
        self.assertEqual(report.inf_count, 0)

    def test_nan_state_reports_bad_tensor(self) -> None:
        report = check_state_finite({"weight": torch.tensor([1.0, float("nan")])})

        self.assertFalse(report.ok)
        self.assertEqual(report.bad_tensor_names, ("weight",))
        self.assertEqual(report.nan_count, 1)
        self.assertEqual(report.inf_count, 0)

    def test_inf_state_reports_bad_tensor(self) -> None:
        report = check_state_finite({"bias": torch.tensor([float("-inf"), 0.0])})

        self.assertFalse(report.ok)
        self.assertEqual(report.bad_tensor_names, ("bias",))
        self.assertEqual(report.nan_count, 0)
        self.assertEqual(report.inf_count, 1)


if __name__ == "__main__":
    unittest.main()
