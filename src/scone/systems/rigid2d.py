from __future__ import annotations

from dataclasses import dataclass

import torch

from scone.engine import System
from scone.state import State


@dataclass(frozen=True)
class Disk2D(System):
    mass_value: float
    radius: float
    inertia_value: float
    gravity: float
    ground_height: float
    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        *,
        mass: float,
        radius: float,
        gravity: float,
        ground_height: float,
        device: torch.device,
        dtype: torch.dtype,
        inertia: float | None = None,
    ) -> None:
        inertia_value = inertia if inertia is not None else 0.5 * mass * radius * radius
        object.__setattr__(self, "mass_value", mass)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "inertia_value", inertia_value)
        object.__setattr__(self, "gravity", gravity)
        object.__setattr__(self, "ground_height", ground_height)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", dtype)

    def mass(self) -> torch.Tensor:
        # Elementwise "diagonal mass": [m, m, I] for (vx, vy, omega)
        m = torch.tensor(self.mass_value, device=self.device, dtype=self.dtype)
        I = torch.tensor(self.inertia_value, device=self.device, dtype=self.dtype)
        return torch.stack([m, m, I]).reshape(1, 3)

    def grad_potential(self, q: torch.Tensor) -> torch.Tensor:
        # q: (n, 3) = (x, y, theta). V = m*g*(y - (ground + radius)) (up to constant).
        m = torch.tensor(self.mass_value, device=q.device, dtype=q.dtype)
        g = torch.tensor(self.gravity, device=q.device, dtype=q.dtype)
        grad = torch.zeros_like(q)
        grad[:, 1] = m * g
        return grad

    def energy(self, state: State) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # state.q/state.v: (n,3) = (x,y,theta)/(vx,vy,omega)
        m = torch.tensor(self.mass_value, device=state.v.device, dtype=state.v.dtype)
        I = torch.tensor(self.inertia_value, device=state.v.device, dtype=state.v.dtype)
        vx = state.v[:, 0]
        vy = state.v[:, 1]
        omega = state.v[:, 2]

        e_kin = 0.5 * m * (vx * vx + vy * vy) + 0.5 * I * (omega * omega)

        y = state.q[:, 1]
        ground = torch.tensor(self.ground_height, device=state.q.device, dtype=state.q.dtype)
        height = y - (ground + torch.tensor(self.radius, device=state.q.device, dtype=state.q.dtype))
        g = torch.tensor(self.gravity, device=state.q.device, dtype=state.q.dtype)
        e_pot = m * g * height

        e_kin_sum = e_kin.sum()
        e_pot_sum = e_pot.sum()
        return e_kin_sum, e_pot_sum, e_kin_sum + e_pot_sum

