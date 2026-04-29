from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class StateFiniteReport:
    ok: bool
    bad_tensor_names: tuple[str, ...]
    nan_count: int
    inf_count: int

    def format_summary(self, *, limit: int = 8) -> str:
        shown_names = ", ".join(self.bad_tensor_names[:limit])
        if len(self.bad_tensor_names) > limit:
            shown_names = f"{shown_names}, ..."
        return (
            f"ok={self.ok}, nan_count={self.nan_count}, inf_count={self.inf_count}, "
            f"bad_tensors=[{shown_names}]"
        )


def extract_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().to("cpu").clone() for name, tensor in model.state_dict().items()}


def load_model_state(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({name: tensor.to(device) for name, tensor in state.items()})


def check_state_finite(state: dict[str, torch.Tensor]) -> StateFiniteReport:
    bad_tensor_names: list[str] = []
    nan_count = 0
    inf_count = 0

    for name, tensor in state.items():
        if not torch.is_floating_point(tensor):
            continue

        tensor_cpu = tensor.detach().to("cpu")
        tensor_nan_count = int(torch.isnan(tensor_cpu).sum().item())
        tensor_inf_count = int(torch.isinf(tensor_cpu).sum().item())
        if tensor_nan_count or tensor_inf_count:
            bad_tensor_names.append(name)
            nan_count += tensor_nan_count
            inf_count += tensor_inf_count

    return StateFiniteReport(
        ok=not bad_tensor_names,
        bad_tensor_names=tuple(bad_tensor_names),
        nan_count=nan_count,
        inf_count=inf_count,
    )


def require_state_finite(state: dict[str, torch.Tensor], *, context: str) -> None:
    report = check_state_finite(state)
    if not report.ok:
        raise ValueError(f"{context} contains non-finite tensors: {report.format_summary()}")


def compute_model_delta(
    current_state: dict[str, torch.Tensor], base_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {name: current_state[name] - base_state[name] for name in current_state}


def add_state_delta(
    base_state: dict[str, torch.Tensor], delta_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {name: base_state[name] + delta_state[name] for name in base_state}


def weighted_average_states(
    states: list[dict[str, torch.Tensor]], weights: list[int]
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states must not be empty")
    if len(states) != len(weights):
        raise ValueError("states and weights must have the same length")

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")

    averaged: dict[str, torch.Tensor] = {}
    for name in states[0]:
        weighted_sum = None
        for state, weight in zip(states, weights):
            contribution = state[name] * float(weight)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        averaged[name] = weighted_sum / float(total_weight)
    return averaged


def digest_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()
