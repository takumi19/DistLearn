from __future__ import annotations

import hashlib

import torch
import torch.nn as nn


def extract_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().to("cpu").clone() for name, tensor in model.state_dict().items()}


def load_model_state(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({name: tensor.to(device) for name, tensor in state.items()})


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
