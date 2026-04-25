from __future__ import annotations

import torch


def select_device(preference: list[str]) -> torch.device:
    for device_name in preference:
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if device_name == "mps" and _mps_available():
            return torch.device("mps")
        if device_name == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def _mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
