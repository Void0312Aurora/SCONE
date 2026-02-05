from __future__ import annotations

from dataclasses import dataclass

import torch

from scone.engine import System
from scone.state import State


@dataclass(frozen=True)
class HarmonicOscillator1D(System):
    mass_value: float
    stiffness: float
    device: torch.device
    dtype: torch.dtype

    def __init__(self, mass: float, stiffness: float, device: torch.device, dtype: torch.dtype) -> None:
        object.__setattr__(self, "mass_value", mass)
        object.__setattr__(self, "stiffness", stiffness)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", dtype)

    def mass(self) -> torch.Tensor:
        return torch.tensor(self.mass_value, device=self.device, dtype=self.dtype)

    def grad_potential(self, q: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.stiffness, device=q.device, dtype=q.dtype) * q

    def energy(self, state: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mass = self.mass()
        e_kin = 0.5 * mass * (state.v * state.v)
        e_pot = 0.5 * torch.tensor(self.stiffness, device=state.q.device, dtype=state.q.dtype) * (state.q * state.q)
        e_kin_sum = e_kin.sum()
        e_pot_sum = e_pot.sum()
        return e_kin_sum, e_pot_sum, e_kin_sum + e_pot_sum


@dataclass(frozen=True)
class BouncingBall1D(System):
    mass_value: float
    gravity: float
    ground_height: float
    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        mass: float,
        gravity: float,
        ground_height: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        object.__setattr__(self, "mass_value", mass)
        object.__setattr__(self, "gravity", gravity)
        object.__setattr__(self, "ground_height", ground_height)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", dtype)

    def mass(self) -> torch.Tensor:
        return torch.tensor(self.mass_value, device=self.device, dtype=self.dtype)

    def grad_potential(self, q: torch.Tensor) -> torch.Tensor:
        # V(q) = m*g*(q - ground_height) for q>=ground (up to a constant shift). grad V = m*g
        mass = self.mass()
        return mass * torch.tensor(self.gravity, device=q.device, dtype=q.dtype)

    def energy(self, state: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mass = self.mass()
        e_kin = 0.5 * mass * (state.v * state.v)
        height = state.q - torch.tensor(self.ground_height, device=state.q.device, dtype=state.q.dtype)
        e_pot = mass * torch.tensor(self.gravity, device=state.q.device, dtype=state.q.dtype) * height
        e_kin_sum = e_kin.sum()
        e_pot_sum = e_pot.sum()
        return e_kin_sum, e_pot_sum, e_kin_sum + e_pot_sum

