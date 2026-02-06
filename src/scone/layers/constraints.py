from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scone.diagnostics import Diagnostics
from scone.engine import ConstraintLayer
from scone.state import State


@dataclass(frozen=True)
class NoOpConstraintLayer(ConstraintLayer):
    def project(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        diagnostics: Diagnostics = {
            "constraints": {
                "residual_pos": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
                "residual_vel": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
                "power_error": torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype),
            }
        }
        return state, diagnostics


def _as_matrix(value: Any, *, cols: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=device, dtype=dtype)
    else:
        value = value.to(device=device, dtype=dtype)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"Constraint matrix must be rank-2, got shape={tuple(value.shape)}")
    if int(value.shape[1]) != int(cols):
        raise ValueError(f"Constraint matrix has wrong width: expected {cols}, got {int(value.shape[1])}")
    return value


def _as_rhs(value: Any, *, rows: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if value is None:
        return torch.zeros((rows,), device=device, dtype=dtype)
    if not isinstance(value, torch.Tensor):
        rhs = torch.as_tensor(value, device=device, dtype=dtype)
    else:
        rhs = value.to(device=device, dtype=dtype)
    rhs = rhs.reshape(-1)
    if rhs.numel() == 1 and rows != 1:
        rhs = rhs.repeat(rows)
    if int(rhs.numel()) != int(rows):
        raise ValueError(f"Constraint rhs has wrong length: expected {rows}, got {int(rhs.numel())}")
    return rhs


def _project_affine(
    *,
    x: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if A.numel() == 0 or A.shape[0] == 0:
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x, zero, zero, torch.zeros((0,), device=x.device, dtype=x.dtype)

    residual_before = A @ x - b
    normal = A @ A.transpose(0, 1)
    eye = torch.eye(int(normal.shape[0]), device=x.device, dtype=x.dtype)
    normal_reg = normal + float(eps) * eye
    multipliers = torch.linalg.solve(normal_reg, residual_before.unsqueeze(-1)).reshape(-1)
    x_proj = x - A.transpose(0, 1) @ multipliers
    residual_after = A @ x_proj - b
    return (
        x_proj,
        torch.linalg.vector_norm(residual_before),
        torch.linalg.vector_norm(residual_after),
        multipliers,
    )


@dataclass(frozen=True)
class LinearConstraintProjectionLayer(ConstraintLayer):
    eps: float = 1e-9
    enforce_position: bool = True
    enforce_velocity: bool = True

    def project(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        q_flat = state.q.reshape(-1)
        v_flat = state.v.reshape(-1)
        device = q_flat.device
        dtype = q_flat.dtype

        constraints_cfg = context.get("constraints", {}) if isinstance(context, dict) else {}
        if not isinstance(constraints_cfg, dict):
            constraints_cfg = {}

        q_next = q_flat
        v_next = v_flat
        residual_pos = torch.tensor(0.0, device=device, dtype=dtype)
        residual_vel = torch.tensor(0.0, device=device, dtype=dtype)
        power_error = torch.tensor(0.0, device=device, dtype=dtype)

        if self.enforce_position:
            A_pos = _as_matrix(
                constraints_cfg.get("A_pos"),
                cols=int(q_flat.numel()),
                device=device,
                dtype=dtype,
            )
            if A_pos is not None:
                b_pos = _as_rhs(
                    constraints_cfg.get("b_pos"),
                    rows=int(A_pos.shape[0]),
                    device=device,
                    dtype=dtype,
                )
                q_next, _, residual_pos, _ = _project_affine(x=q_next, A=A_pos, b=b_pos, eps=self.eps)

        if self.enforce_velocity:
            A_vel = _as_matrix(
                constraints_cfg.get("A_vel"),
                cols=int(v_flat.numel()),
                device=device,
                dtype=dtype,
            )
            if A_vel is not None:
                b_vel = _as_rhs(
                    constraints_cfg.get("b_vel"),
                    rows=int(A_vel.shape[0]),
                    device=device,
                    dtype=dtype,
                )
                v_next, _, residual_vel, multipliers_vel = _project_affine(x=v_next, A=A_vel, b=b_vel, eps=self.eps)
                post_residual = A_vel @ v_next - b_vel
                power_error = torch.abs(torch.dot(multipliers_vel, post_residual)) / torch.tensor(
                    max(dt, 1e-12), device=device, dtype=dtype
                )

        next_state = State(q=q_next.reshape_as(state.q), v=v_next.reshape_as(state.v), t=state.t)
        diagnostics: Diagnostics = {
            "constraints": {
                "residual_pos": residual_pos,
                "residual_vel": residual_vel,
                "power_error": power_error,
            }
        }
        return next_state, diagnostics
