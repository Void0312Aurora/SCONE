from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scone.contacts import Contact, ContactMode, contact_mode_name
from scone.diagnostics import Diagnostics
from scone.engine import EventLayer
from scone.sleep import SleepManager
from scone.state import State


def _as_tensor(value: float | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _perp2(v: torch.Tensor) -> torch.Tensor:
    return torch.stack([-v[1], v[0]])


def _cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[0] * b[1] - a[1] * b[0]


def _omega_cross_r(omega: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return torch.stack([-omega * r[1], omega * r[0]])


def _index_add_row(x: torch.Tensor, *, index: int, value: torch.Tensor) -> torch.Tensor:
    idx = torch.tensor([int(index)], device=x.device, dtype=torch.long)
    return torch.index_add(x, 0, idx, value.unsqueeze(0))


def _index_add_two_rows(
    x: torch.Tensor, *, index0: int, value0: torch.Tensor, index1: int | None, value1: torch.Tensor | None
) -> torch.Tensor:
    if index1 is None or value1 is None:
        return _index_add_row(x, index=index0, value=value0)
    idx = torch.tensor([int(index0), int(index1)], device=x.device, dtype=torch.long)
    src = torch.stack([value0, value1], dim=0)
    return torch.index_add(x, 0, idx, src)


@dataclass(frozen=True)
class NoOpEventLayer(EventLayer):
    def resolve(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        zero = torch.tensor(0.0, device=state.q.device, dtype=state.q.dtype)
        diagnostics: Diagnostics = {
            "contacts": {
                "penetration_max": zero,
                "gap_max": zero,
                "complementarity_residual_max": zero,
                "items": [],
            }
        }
        return state, diagnostics


@dataclass(frozen=True)
class BouncingBallEventLayer(EventLayer):
    mass: float
    gravity: float
    restitution: float
    ground_height: float = 0.0
    contact_slop: float = 1e-3
    impact_velocity_min: float = 0.1
    sleep: SleepManager | None = None

    def resolve(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        q = state.q
        v = state.v
        ground = torch.tensor(self.ground_height, device=q.device, dtype=q.dtype)
        mass = torch.tensor(self.mass, device=q.device, dtype=q.dtype)
        gravity = torch.tensor(self.gravity, device=q.device, dtype=q.dtype)
        slop = torch.tensor(self.contact_slop, device=q.device, dtype=q.dtype)
        v_impact_min = torch.tensor(self.impact_velocity_min, device=q.device, dtype=q.dtype)

        phi = q - ground
        penetration = torch.clamp(-phi, min=0.0)
        contact_active = phi <= slop
        is_penetrating = phi < 0
        do_impact = is_penetrating & (v < -v_impact_min)

        # Position correction (projection) for penetration only.
        next_q = torch.where(is_penetrating, ground, q)
        next_v = torch.where(do_impact, -self.restitution * v, v)
        base_state = State(q=next_q, v=next_v, t=state.t)

        contacts: list[dict[str, Any]] = []
        if contact_active.ndim == 0:
            active_indices = [0] if bool(contact_active.item()) else []
        else:
            active_indices = torch.nonzero(contact_active).flatten().tolist()

        for idx in active_indices:
            idx_int = int(idx)
            mode = int(ContactMode.IMPACT) if bool(do_impact.flatten()[idx_int].item()) else int(ContactMode.ACTIVE)
            n = torch.ones((1,), device=q.device, dtype=q.dtype)
            contact = Contact(
                id=f"body{idx_int}-ground",
                body_i=idx_int,
                body_j=-1,
                phi=phi.flatten()[idx_int],
                n=n,
                lambda_n=torch.tensor(0.0, device=q.device, dtype=q.dtype),
                lambda_t=torch.zeros((0,), device=q.device, dtype=q.dtype),
                mode=mode,
            )
            contact_dict = contact.to_dict()
            contact_dict["phi_next"] = (next_q.flatten()[idx_int] - ground).detach()
            contacts.append(contact_dict)

        next_state = base_state
        sleep_diag: Diagnostics = {}
        if self.sleep is not None:
            next_state, contacts, sleep_diag = self.sleep.apply(state=next_state, contacts=contacts, context=context)

        # Fill impulses and event ledger using the *final* state after sleep.
        delta_v = next_state.v - v
        impulse = mass * delta_v

        for c in contacts:
            body_i = int(c.get("body_i", 0))
            if v.ndim == 0:
                c["lambda_n"] = impulse.reshape(1)[0]
            else:
                c["lambda_n"] = impulse.flatten()[body_i]
            c.setdefault("lambda_t", torch.zeros((0,), device=q.device, dtype=q.dtype))
            c["mode_name"] = contact_mode_name(int(c.get("mode", 0)))

        # Discrete work via kinetic energy jump: ΔT = 0.5 * m * (v+^2 - v-^2) = 0.5 * (v+ + v-) * I
        d_ke = 0.5 * mass * (next_state.v * next_state.v - v * v)
        w_impulse = 0.5 * (next_state.v + v) * impulse
        d_pe = mass * gravity * (next_state.q - q)
        d_e_event = d_ke + d_pe

        phi_next = next_state.q - ground
        gap = torch.clamp(phi_next, min=0.0)
        effective_gap = torch.clamp(gap - slop, min=0.0)
        if bool(contact_active.any().item()) if contact_active.ndim > 0 else bool(contact_active.item()):
            gap_max = gap[contact_active].max() if contact_active.ndim > 0 else gap.reshape(1)[0]
            complementarity_residual_max = torch.abs(torch.clamp(impulse, min=0.0) * effective_gap)[
                contact_active
            ].max()
        else:
            gap_max = torch.tensor(0.0, device=q.device, dtype=q.dtype)
            complementarity_residual_max = torch.tensor(0.0, device=q.device, dtype=q.dtype)

        diagnostics: Diagnostics = {
            "contacts": {
                "penetration_max": penetration.max(),
                "gap_max": gap_max,
                "complementarity_residual_max": complementarity_residual_max,
                "items": contacts,
            },
            "event": {
                "W_impulse": w_impulse.sum(),
                "W_position_correction": d_pe.sum(),
                "dE_kin": d_ke.sum(),
                "dE_pot": d_pe.sum(),
                "dE_event": d_e_event.sum(),
            },
        }
        diagnostics.update(sleep_diag)
        return next_state, diagnostics


@dataclass(frozen=True)
class DiskGroundContactEventLayer(EventLayer):
    mass: float
    inertia: float
    radius: float
    gravity: float
    friction_mu: float | torch.Tensor = 0.5
    restitution: float = 0.0
    ground_height: float = 0.0
    contact_slop: float = 1e-3
    impact_velocity_min: float = 0.1
    sleep: SleepManager | None = None

    def resolve(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        q = state.q
        v = state.v
        device = q.device
        dtype = q.dtype

        if q.ndim != 2 or q.shape[-1] != 3 or v.ndim != 2 or v.shape[-1] != 3:
            raise ValueError("DiskGroundContactEventLayer expects state.q/state.v of shape (n,3)")

        ground = torch.tensor(self.ground_height, device=device, dtype=dtype)
        radius = torch.tensor(self.radius, device=device, dtype=dtype)
        m = torch.tensor(self.mass, device=device, dtype=dtype)
        I = torch.tensor(self.inertia, device=device, dtype=dtype)
        g = torch.tensor(self.gravity, device=device, dtype=dtype)
        mu = _as_tensor(self.friction_mu, device=device, dtype=dtype)
        slop = torch.tensor(self.contact_slop, device=device, dtype=dtype)
        v_impact_min = torch.tensor(self.impact_velocity_min, device=device, dtype=dtype)

        y = q[:, 1]
        phi = y - (ground + radius)
        penetration = torch.clamp(-phi, min=0.0)
        contact_active = phi <= slop
        is_penetrating = phi < 0

        vx = v[:, 0]
        vy = v[:, 1]
        omega = v[:, 2]

        do_impact = is_penetrating & (vy < -v_impact_min)
        need_normal = contact_active & (vy < 0)

        vy_target = torch.where(do_impact, -self.restitution * vy, torch.zeros_like(vy))
        jn = torch.where(need_normal, m * (vy_target - vy), torch.zeros_like(vy))
        jn = torch.clamp(jn, min=0.0)

        denom = (1.0 / m) + (radius * radius) / I
        vt = vx + omega * radius
        jt_stick = -vt / denom
        jt_limit = mu * jn
        jt = torch.where(torch.abs(jt_stick) <= jt_limit, jt_stick, -jt_limit * torch.sign(vt))
        jt = torch.where(jn > 0, jt, torch.zeros_like(jt))

        next_vx = vx + jt / m
        next_vy = vy + jn / m
        next_omega = omega + (radius * jt) / I

        next_x = q[:, 0]
        next_y = torch.where(is_penetrating, ground + radius, y)
        next_theta = q[:, 2]
        base_state = State(
            q=torch.stack([next_x, next_y, next_theta], dim=-1),
            v=torch.stack([next_vx, next_vy, next_omega], dim=-1),
            t=state.t,
        )

        # Build contacts (placeholders for lambdas; sleep may modify modes and velocities).
        contacts: list[dict[str, Any]] = []
        active_indices = torch.nonzero(contact_active).flatten().tolist()
        for idx in active_indices:
            i = int(idx)
            mode: int
            if bool(do_impact[i].item()):
                mode = int(ContactMode.IMPACT)
            elif bool(jn[i].item() == 0.0) and bool(contact_active[i].item()):
                mode = int(ContactMode.ACTIVE)
            else:
                is_sticking = bool(torch.abs(jt_stick[i]).item() <= jt_limit[i].item() + 1e-12)
                mode = int(ContactMode.STICKING if is_sticking else ContactMode.SLIDING)

            n = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
            contact = Contact(
                id=f"disk{i}-ground",
                body_i=i,
                body_j=-1,
                phi=phi[i],
                n=n,
                lambda_n=torch.tensor(0.0, device=device, dtype=dtype),
                lambda_t=torch.zeros((1,), device=device, dtype=dtype),
                mode=mode,
            )
            c = contact.to_dict()
            c["phi_next"] = (base_state.q[i, 1] - (ground + radius)).detach()
            contacts.append(c)

        next_state = base_state
        sleep_diag: Diagnostics = {}
        if self.sleep is not None:
            next_state, contacts, sleep_diag = self.sleep.apply(state=next_state, contacts=contacts, context=context)

        # Populate impulses based on final delta-v (includes sleep effects).
        delta_v = next_state.v - v
        lambda_n = m * delta_v[:, 1]
        lambda_t = m * delta_v[:, 0]
        lambda_t_y = torch.tensor(0.0, device=device, dtype=dtype)

        for c in contacts:
            body_i = int(c.get("body_i", 0))
            c["lambda_n"] = lambda_n[body_i]
            c["lambda_t"] = torch.stack([lambda_t[body_i], lambda_t_y])
            c["mode_name"] = contact_mode_name(int(c.get("mode", 0)))

        # Event ledger (kinetic + potential jump).
        d_ke = 0.5 * m * ((next_state.v[:, 0] ** 2 + next_state.v[:, 1] ** 2) - (vx * vx + vy * vy)) + 0.5 * I * (
            (next_state.v[:, 2] ** 2) - (omega * omega)
        )
        d_pe = m * g * (next_state.q[:, 1] - y)
        d_e_event = d_ke + d_pe

        phi_next = next_state.q[:, 1] - (ground + radius)
        gap = torch.clamp(phi_next, min=0.0)
        effective_gap = torch.clamp(gap - slop, min=0.0)
        if bool(contact_active.any().item()):
            gap_max = gap[contact_active].max()
            complementarity_residual_max = torch.abs(torch.clamp(lambda_n, min=0.0) * effective_gap)[contact_active].max()
        else:
            gap_max = torch.tensor(0.0, device=device, dtype=dtype)
            complementarity_residual_max = torch.tensor(0.0, device=device, dtype=dtype)

        diagnostics: Diagnostics = {
            "contacts": {
                "penetration_max": penetration.max(),
                "gap_max": gap_max,
                "complementarity_residual_max": complementarity_residual_max,
                "items": contacts,
            },
            "event": {
                "W_impulse": d_ke.sum(),
                "W_position_correction": d_pe.sum(),
                "dE_kin": d_ke.sum(),
                "dE_pot": d_pe.sum(),
                "dE_event": d_e_event.sum(),
            },
        }
        diagnostics.update(sleep_diag)
        return next_state, diagnostics


@dataclass(frozen=True)
class DiskContactPGSEventLayer(EventLayer):
    mass: float
    inertia: float
    radius: float
    gravity: float
    friction_mu: float | torch.Tensor = 0.5
    friction_mu_pair: float | torch.Tensor | None = None
    restitution: float = 0.0
    ground_height: float = 0.0
    contact_slop: float = 1e-3
    impact_velocity_min: float = 0.1
    pgs_iters: int = 20
    baumgarte_beta: float = 0.2
    residual_tol: float = 1e-6
    warm_start: bool = True
    sleep: SleepManager | None = None

    def resolve(self, state: State, dt: float, context: dict[str, Any]) -> tuple[State, Diagnostics]:
        q = state.q
        v = state.v
        device = q.device
        dtype = q.dtype

        if q.ndim != 2 or q.shape[-1] != 3 or v.ndim != 2 or v.shape[-1] != 3:
            raise ValueError("DiskContactPGSEventLayer expects state.q/state.v of shape (n,3)")

        n_bodies = int(q.shape[0])
        if n_bodies == 0:
            diagnostics: Diagnostics = {
                "contacts": {
                    "penetration_max": torch.tensor(0.0, device=device, dtype=dtype),
                    "gap_max": torch.tensor(0.0, device=device, dtype=dtype),
                    "complementarity_residual_max": torch.tensor(0.0, device=device, dtype=dtype),
                    "items": [],
                }
            }
            return state, diagnostics

        ground = torch.tensor(self.ground_height, device=device, dtype=dtype)
        radius = torch.tensor(self.radius, device=device, dtype=dtype)
        m = torch.tensor(self.mass, device=device, dtype=dtype)
        inv_m = 1.0 / m
        I = torch.tensor(self.inertia, device=device, dtype=dtype)
        inv_I = 1.0 / I
        g = torch.tensor(self.gravity, device=device, dtype=dtype)
        mu_ground = _as_tensor(self.friction_mu, device=device, dtype=dtype)
        mu_pair = mu_ground if self.friction_mu_pair is None else _as_tensor(self.friction_mu_pair, device=device, dtype=dtype)
        slop = torch.tensor(self.contact_slop, device=device, dtype=dtype)
        v_impact_min = torch.tensor(self.impact_velocity_min, device=device, dtype=dtype)
        beta = float(self.baumgarte_beta)

        pos = q[:, :2]
        vel = v[:, :2].clone()
        omega = v[:, 2].clone()

        contacts_work: list[dict[str, Any]] = []
        sleeping_mask = None
        if isinstance(context, dict):
            sleep_ctx = context.get("sleep", {})
            if isinstance(sleep_ctx, dict):
                mask = sleep_ctx.get("sleeping_mask")
                if isinstance(mask, torch.Tensor) and mask.shape == (n_bodies,):
                    sleeping_mask = mask.to(device=device)

        ground_y = ground + radius
        for body_i in range(n_bodies):
            phi = pos[body_i, 1] - ground_y
            if bool((phi <= slop).item()):
                n = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
                t = _perp2(n)
                r_i = torch.stack([torch.tensor(0.0, device=device, dtype=dtype), -radius])
                r_j = torch.zeros((2,), device=device, dtype=dtype)
                v_rel = vel[body_i] + _omega_cross_r(omega[body_i], r_i)
                vn0 = (v_rel * n).sum()
                impact = bool((phi < 0).item() and (vn0 < -v_impact_min).item())
                contacts_work.append(
                    {
                        "id": f"disk{body_i}-ground",
                        "body_i": body_i,
                        "body_j": -1,
                        "phi": phi,
                        "n": n,
                        "t": t,
                        "r_i": r_i,
                        "r_j": r_j,
                        "vn0": vn0,
                        "impact": impact,
                        "mu": mu_ground,
                        "lambda_n": torch.tensor(0.0, device=device, dtype=dtype),
                        "lambda_t": torch.tensor(0.0, device=device, dtype=dtype),
                    }
                )

        for body_i in range(n_bodies):
            for body_j in range(body_i + 1, n_bodies):
                delta = pos[body_i] - pos[body_j]
                dist = torch.linalg.vector_norm(delta)
                if bool((dist <= 1e-12).item()):
                    n = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
                else:
                    n = delta / dist
                phi = dist - (radius + radius)
                if not bool((phi <= slop).item()):
                    continue

                t = _perp2(n)
                r_i = -n * radius
                r_j = n * radius
                v_i = vel[body_i] + _omega_cross_r(omega[body_i], r_i)
                v_j = vel[body_j] + _omega_cross_r(omega[body_j], r_j)
                v_rel = v_i - v_j
                vn0 = (v_rel * n).sum()
                impact = bool((phi < 0).item() and (vn0 < -v_impact_min).item())
                contacts_work.append(
                    {
                        "id": f"disk{body_i}-disk{body_j}",
                        "body_i": body_i,
                        "body_j": body_j,
                        "phi": phi,
                        "n": n,
                        "t": t,
                        "r_i": r_i,
                        "r_j": r_j,
                        "vn0": vn0,
                        "impact": impact,
                        "mu": mu_pair,
                        "lambda_n": torch.tensor(0.0, device=device, dtype=dtype),
                        "lambda_t": torch.tensor(0.0, device=device, dtype=dtype),
                    }
                )

        if not contacts_work:
            diagnostics: Diagnostics = {
                "contacts": {
                    "penetration_max": torch.tensor(0.0, device=device, dtype=dtype),
                    "gap_max": torch.tensor(0.0, device=device, dtype=dtype),
                    "complementarity_residual_max": torch.tensor(0.0, device=device, dtype=dtype),
                    "items": [],
                },
                "event": {
                    "W_impulse": torch.tensor(0.0, device=device, dtype=dtype),
                    "W_position_correction": torch.tensor(0.0, device=device, dtype=dtype),
                    "dE_kin": torch.tensor(0.0, device=device, dtype=dtype),
                    "dE_pot": torch.tensor(0.0, device=device, dtype=dtype),
                    "dE_event": torch.tensor(0.0, device=device, dtype=dtype),
                },
            }
            return state, diagnostics

        cache: dict[str, Any] | None = None
        if self.warm_start and isinstance(context, dict):
            cached = context.setdefault("pgs_cache", {})
            if isinstance(cached, dict):
                cache = cached

        if cache is not None:
            for c in contacts_work:
                body_i = int(c["body_i"])
                body_j = int(c["body_j"])
                if sleeping_mask is not None and (
                    (body_i >= 0 and bool(sleeping_mask[body_i].item()))
                    or (body_j >= 0 and bool(sleeping_mask[body_j].item()))
                ):
                    continue

                cached = cache.get(str(c["id"]))
                if not isinstance(cached, dict):
                    continue
                lambda_n = cached.get("lambda_n")
                lambda_t = cached.get("lambda_t")
                if not isinstance(lambda_n, torch.Tensor):
                    lambda_n = torch.tensor(float(lambda_n), device=device, dtype=dtype)
                if not isinstance(lambda_t, torch.Tensor):
                    lambda_t = torch.tensor(float(lambda_t), device=device, dtype=dtype)

                lambda_n = torch.clamp(lambda_n.to(device=device, dtype=dtype), min=0.0)
                lambda_t = lambda_t.to(device=device, dtype=dtype)
                mu_c = c["mu"]
                friction_limit = mu_c * lambda_n
                lambda_t = torch.clamp(lambda_t, min=-friction_limit, max=friction_limit)

                c["lambda_n"] = lambda_n
                c["lambda_t"] = lambda_t

                impulse_n = lambda_n * c["n"]
                impulse_t = lambda_t * c["t"]
                impulse = impulse_n + impulse_t

                delta_vel_i = impulse * inv_m
                if body_j >= 0:
                    delta_vel_j = -impulse * inv_m
                    vel = _index_add_two_rows(
                        vel, index0=body_i, value0=delta_vel_i, index1=body_j, value1=delta_vel_j
                    )
                else:
                    vel = _index_add_row(vel, index=body_i, value=delta_vel_i)

                delta_omega_i = _cross2(c["r_i"], impulse_t) * inv_I
                if body_j >= 0:
                    delta_omega_j = _cross2(c["r_j"], -impulse_t) * inv_I
                    omega = _index_add_two_rows(
                        omega, index0=body_i, value0=delta_omega_i, index1=body_j, value1=delta_omega_j
                    )
                else:
                    omega = _index_add_row(omega, index=body_i, value=delta_omega_i)

        pgs_iters = int(max(1, self.pgs_iters))
        residual_tol = torch.tensor(self.residual_tol, device=device, dtype=dtype)
        status = "max_iter"
        residual_max = torch.tensor(float("inf"), device=device, dtype=dtype)
        iters_done = 0

        for iter_index in range(pgs_iters):
            for c in contacts_work:
                body_i = int(c["body_i"])
                body_j = int(c["body_j"])
                n = c["n"]
                t = c["t"]
                r_i = c["r_i"]
                r_j = c["r_j"]

                v_i = vel[body_i] + _omega_cross_r(omega[body_i], r_i)
                if body_j >= 0:
                    v_j = vel[body_j] + _omega_cross_r(omega[body_j], r_j)
                else:
                    v_j = torch.zeros((2,), device=device, dtype=dtype)

                v_rel = v_i - v_j
                vn = (v_rel * n).sum()
                vt = (v_rel * t).sum()

                inv_m_i = inv_m
                inv_m_j = inv_m if body_j >= 0 else torch.tensor(0.0, device=device, dtype=dtype)

                k_n = inv_m_i + inv_m_j
                penetration = torch.clamp(-(c["phi"] + slop), min=0.0)
                bias = torch.tensor(beta, device=device, dtype=dtype) * penetration / torch.tensor(
                    dt, device=device, dtype=dtype
                )
                restitution_target = torch.tensor(0.0, device=device, dtype=dtype)
                if c["impact"]:
                    restitution_target = -torch.tensor(self.restitution, device=device, dtype=dtype) * c["vn0"]

                vn_target = bias + restitution_target
                delta_lambda_n = (vn_target - vn) / k_n

                lambda_n_old = c["lambda_n"]
                lambda_n_new = torch.clamp(lambda_n_old + delta_lambda_n, min=0.0)
                delta_lambda_n = lambda_n_new - lambda_n_old
                c["lambda_n"] = lambda_n_new

                impulse_n = delta_lambda_n * n
                delta_vel_i = impulse_n * inv_m_i
                if body_j >= 0:
                    delta_vel_j = -impulse_n * inv_m_j
                    vel = _index_add_two_rows(vel, index0=body_i, value0=delta_vel_i, index1=body_j, value1=delta_vel_j)
                else:
                    vel = _index_add_row(vel, index=body_i, value=delta_vel_i)

                k_t = inv_m_i + inv_m_j
                r_i_cross_t = _cross2(r_i, t)
                k_t = k_t + (r_i_cross_t * r_i_cross_t) * inv_I
                if body_j >= 0:
                    r_j_cross_t = _cross2(r_j, t)
                    k_t = k_t + (r_j_cross_t * r_j_cross_t) * inv_I

                delta_lambda_t = (-vt) / k_t
                lambda_t_old = c["lambda_t"]
                lambda_t_candidate = lambda_t_old + delta_lambda_t
                mu_c = c["mu"]
                friction_limit = mu_c * c["lambda_n"]
                lambda_t_new = torch.clamp(lambda_t_candidate, min=-friction_limit, max=friction_limit)
                delta_lambda_t = lambda_t_new - lambda_t_old
                c["lambda_t"] = lambda_t_new

                impulse_t = delta_lambda_t * t
                delta_vel_i = impulse_t * inv_m_i
                if body_j >= 0:
                    delta_vel_j = -impulse_t * inv_m_j
                    vel = _index_add_two_rows(vel, index0=body_i, value0=delta_vel_i, index1=body_j, value1=delta_vel_j)
                else:
                    vel = _index_add_row(vel, index=body_i, value=delta_vel_i)

                delta_omega_i = _cross2(r_i, impulse_t) * inv_I
                if body_j >= 0:
                    delta_omega_j = _cross2(r_j, -impulse_t) * inv_I
                    omega = _index_add_two_rows(
                        omega, index0=body_i, value0=delta_omega_i, index1=body_j, value1=delta_omega_j
                    )
                else:
                    omega = _index_add_row(omega, index=body_i, value=delta_omega_i)

            residual_max = torch.tensor(0.0, device=device, dtype=dtype)
            for c in contacts_work:
                body_i = int(c["body_i"])
                body_j = int(c["body_j"])
                n = c["n"]
                t = c["t"]
                r_i = c["r_i"]
                r_j = c["r_j"]

                v_i = vel[body_i] + _omega_cross_r(omega[body_i], r_i)
                if body_j >= 0:
                    v_j = vel[body_j] + _omega_cross_r(omega[body_j], r_j)
                else:
                    v_j = torch.zeros((2,), device=device, dtype=dtype)

                v_rel = v_i - v_j
                vn = (v_rel * n).sum()
                vt = (v_rel * t).sum()

                penetration = torch.clamp(-(c["phi"] + slop), min=0.0)
                bias = torch.tensor(beta, device=device, dtype=dtype) * penetration / torch.tensor(
                    dt, device=device, dtype=dtype
                )
                restitution_target = torch.tensor(0.0, device=device, dtype=dtype)
                if c["impact"]:
                    restitution_target = -torch.tensor(self.restitution, device=device, dtype=dtype) * c["vn0"]
                vn_target = bias + restitution_target

                lambda_n = c["lambda_n"]
                lambda_t = c["lambda_t"]
                mu_c = c["mu"]
                friction_limit = mu_c * lambda_n

                if bool((lambda_n > 0.0).item()):
                    res_n = torch.abs(vn - vn_target)
                else:
                    res_n = torch.clamp(vn_target - vn, min=0.0)

                bound_violation = torch.clamp(lambda_t.abs() - friction_limit, min=0.0)
                stick_threshold = friction_limit - torch.tensor(1e-12, device=device, dtype=dtype)
                sticking = bool((lambda_t.abs() < stick_threshold).item())
                if sticking:
                    res_t = torch.abs(vt)
                else:
                    res_t = torch.clamp(lambda_t * vt, min=0.0)

                residual = torch.maximum(res_n, torch.maximum(res_t, bound_violation))
                residual_max = torch.maximum(residual_max, residual)

            iters_done = iter_index + 1
            if bool((residual_max <= residual_tol).item()):
                status = "converged"
                break

        pos_next = pos.clone()
        for c in contacts_work:
            body_i = int(c["body_i"])
            body_j = int(c["body_j"])

            if body_j < 0:
                phi = pos_next[body_i, 1] - ground_y
                if bool((phi < -slop).item()):
                    correction = -(phi + slop)
                    delta_pos = torch.stack([torch.zeros_like(correction), correction])
                    pos_next = _index_add_row(pos_next, index=body_i, value=delta_pos)
                continue

            delta = pos_next[body_i] - pos_next[body_j]
            dist = torch.linalg.vector_norm(delta)
            if bool((dist <= 1e-12).item()):
                n = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
                dist = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                n = delta / dist
            phi = dist - (radius + radius)
            if bool((phi < -slop).item()):
                correction = -(phi + slop)
                weight_i = inv_m
                weight_j = inv_m
                total = weight_i + weight_j
                delta_i = n * correction * (weight_i / total)
                delta_j = -n * correction * (weight_j / total)
                pos_next = _index_add_two_rows(pos_next, index0=body_i, value0=delta_i, index1=body_j, value1=delta_j)

        next_q = torch.cat([pos_next, q[:, 2:3]], dim=-1)
        next_v = torch.cat([vel, omega.unsqueeze(-1)], dim=-1)
        base_state = State(q=next_q, v=next_v, t=state.t)

        contacts_items: list[dict[str, Any]] = []
        for c in contacts_work:
            body_i = int(c["body_i"])
            body_j = int(c["body_j"])
            phi_next: torch.Tensor
            if body_j < 0:
                phi_next = pos_next[body_i, 1] - ground_y
            else:
                delta = pos_next[body_i] - pos_next[body_j]
                dist = torch.linalg.vector_norm(delta)
                phi_next = dist - (radius + radius)

            lambda_t_vec = c["lambda_t"] * c["t"]

            if c["impact"]:
                mode = int(ContactMode.IMPACT)
            elif bool((c["lambda_n"] <= 0.0).item()):
                mode = int(ContactMode.ACTIVE)
            else:
                mu_c = c["mu"]
                if bool((mu_c <= 0.0).item()):
                    mode = int(ContactMode.ACTIVE)
                else:
                    friction_limit = mu_c * c["lambda_n"]
                    is_sliding = bool((c["lambda_t"].abs() >= (friction_limit - 1e-12)).item())
                    mode = int(ContactMode.SLIDING if is_sliding else ContactMode.STICKING)

            contact = Contact(
                id=str(c["id"]),
                body_i=body_i,
                body_j=body_j,
                phi=c["phi"],
                n=c["n"],
                lambda_n=c["lambda_n"],
                lambda_t=lambda_t_vec,
                mode=mode,
            )
            item = contact.to_dict()
            item["phi_next"] = phi_next.detach()
            item["mu"] = c["mu"].detach() if isinstance(c.get("mu"), torch.Tensor) else c.get("mu")
            contacts_items.append(item)

        next_state = base_state
        sleep_diag: Diagnostics = {}
        if self.sleep is not None:
            next_state, contacts_items, sleep_diag = self.sleep.apply(
                state=next_state, contacts=contacts_items, context=context
            )

        if cache is not None:
            current_ids = {str(c["id"]) for c in contacts_work}
            for stale in list(cache.keys()):
                if stale not in current_ids:
                    del cache[stale]
            for c in contacts_work:
                cache[str(c["id"])] = {
                    "lambda_n": c["lambda_n"].detach(),
                    "lambda_t": c["lambda_t"].detach(),
                }

        vx0 = v[:, 0]
        vy0 = v[:, 1]
        omega0 = v[:, 2]
        vx1 = next_state.v[:, 0]
        vy1 = next_state.v[:, 1]
        omega1 = next_state.v[:, 2]

        d_ke = 0.5 * m * ((vx1 * vx1 + vy1 * vy1) - (vx0 * vx0 + vy0 * vy0)) + 0.5 * I * (
            (omega1 * omega1) - (omega0 * omega0)
        )
        d_pe = m * g * (next_state.q[:, 1] - q[:, 1])
        d_e_event = d_ke + d_pe

        penetration_max = torch.tensor(0.0, device=device, dtype=dtype)
        for c in contacts_work:
            penetration_max = torch.maximum(penetration_max, torch.clamp(-c["phi"], min=0.0))

        gap_max = torch.tensor(0.0, device=device, dtype=dtype)
        complementarity_residual_max = torch.tensor(0.0, device=device, dtype=dtype)
        for item in contacts_items:
            phi_next = item.get("phi_next")
            if not isinstance(phi_next, torch.Tensor):
                continue
            gap = torch.clamp(phi_next, min=0.0)
            gap_max = torch.maximum(gap_max, gap)

            lambda_n = item.get("lambda_n")
            if isinstance(lambda_n, torch.Tensor):
                effective_gap = torch.clamp(gap - slop, min=0.0)
                residual = torch.abs(torch.clamp(lambda_n, min=0.0) * effective_gap)
                complementarity_residual_max = torch.maximum(complementarity_residual_max, residual)

        mode_counts: dict[str, int] = {}
        for item in contacts_items:
            name = str(item.get("mode_name", "unknown"))
            mode_counts[name] = mode_counts.get(name, 0) + 1

        diagnostics: Diagnostics = {
            "contacts": {
                "penetration_max": penetration_max,
                "gap_max": gap_max,
                "complementarity_residual_max": complementarity_residual_max,
                "items": contacts_items,
                "mode_counts": mode_counts,
            },
            "solver": {
                "iters": iters_done,
                "residual_max": residual_max,
                "status": status,
            },
            "event": {
                "W_impulse": d_ke.sum(),
                "W_position_correction": d_pe.sum(),
                "dE_kin": d_ke.sum(),
                "dE_pot": d_pe.sum(),
                "dE_event": d_e_event.sum(),
            },
        }
        diagnostics.update(sleep_diag)
        return next_state, diagnostics
