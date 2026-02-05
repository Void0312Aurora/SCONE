from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scone.contacts import ContactMode, contact_mode_name
from scone.state import State


@dataclass(frozen=True)
class SleepConfig:
    enabled: bool = True
    v_sleep: float = 0.1
    v_wake: float = 0.2
    steps_to_sleep: int = 1
    freeze_core: bool = True


class SleepManager:
    def __init__(self, config: SleepConfig) -> None:
        self._config = config

    def apply(
        self,
        *,
        state: State,
        contacts: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> tuple[State, list[dict[str, Any]], dict[str, Any]]:
        if not self._config.enabled:
            _ensure_sleep_context(context=context, state=state, sleeping_mask=None, freeze_core=self._config.freeze_core)
            diagnostics: dict[str, Any] = {"sleep": {"enabled": False}}
            return state, contacts, diagnostics

        speeds = _speed_per_body(state.v)
        n_bodies = int(speeds.shape[0])

        sleeping, counters = _get_or_init_sleep_state(context=context, n_bodies=n_bodies, device=state.v.device)

        # Wake-up: if a sleeping body is moving fast or experiences an impact contact, wake the entire island.
        wake_mask = sleeping & (speeds > float(self._config.v_wake))
        impacted_bodies = _bodies_with_mode(contacts, mode=int(ContactMode.IMPACT), n_bodies=n_bodies).to(
            device=state.v.device
        )
        wake_mask = wake_mask | (sleeping & impacted_bodies)

        if bool(wake_mask.any().item()):
            sleeping = sleeping & (~wake_mask)
            counters = torch.where(wake_mask, torch.zeros_like(counters), counters)

        islands = _build_islands(contacts, n_bodies=n_bodies)
        supported_by_static = _supported_by_static(contacts, n_bodies=n_bodies).to(device=state.v.device)

        sleep_steps = int(max(1, self._config.steps_to_sleep))
        for island in islands:
            island_mask = torch.zeros((n_bodies,), device=state.v.device, dtype=torch.bool)
            island_mask[island] = True

            if bool((sleeping & island_mask).all().item()):
                continue

            supported = bool((supported_by_static & island_mask).any().item())
            max_speed = speeds[island].max()
            eligible = supported and bool((max_speed <= float(self._config.v_sleep)).item())

            if eligible:
                counters = torch.where(island_mask, counters + 1, counters)
                island_ready = bool((counters[island] >= sleep_steps).all().item())
                if island_ready:
                    sleeping = sleeping | island_mask
            else:
                counters = torch.where(island_mask, torch.zeros_like(counters), counters)

        sleeping_mask = sleeping
        next_state = _apply_sleep_to_state(state, sleeping_mask)
        next_contacts = _apply_sleep_to_contacts(contacts, sleeping_mask)

        _set_sleep_state(context=context, sleeping=sleeping, counters=counters, sleeping_mask=sleeping_mask)
        _ensure_sleep_context(
            context=context, state=state, sleeping_mask=sleeping_mask, freeze_core=self._config.freeze_core
        )

        diagnostics: dict[str, Any] = {
            "sleep": {
                "enabled": True,
                "sleeping_count": int(sleeping_mask.sum().detach().cpu().item()),
                "sleeping_mask": sleeping_mask,
            }
        }
        return next_state, next_contacts, diagnostics


def _speed_per_body(v: torch.Tensor) -> torch.Tensor:
    if v.ndim == 0:
        return v.abs().reshape(1)
    if v.ndim == 1:
        return v.abs()
    return v.norm(dim=-1)


def _get_or_init_sleep_state(
    *, context: dict[str, Any], n_bodies: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    sleep_ctx = context.setdefault("sleep", {})
    state = sleep_ctx.get("state")
    if isinstance(state, dict):
        sleeping = state.get("sleeping")
        counters = state.get("counters")
        if (
            isinstance(sleeping, torch.Tensor)
            and isinstance(counters, torch.Tensor)
            and sleeping.shape == (n_bodies,)
            and counters.shape == (n_bodies,)
        ):
            if sleeping.device != device:
                sleeping = sleeping.to(device)
            if counters.device != device:
                counters = counters.to(device)
            return sleeping, counters

    sleeping = torch.zeros((n_bodies,), device=device, dtype=torch.bool)
    counters = torch.zeros((n_bodies,), device=device, dtype=torch.int64)
    _set_sleep_state(context=context, sleeping=sleeping, counters=counters, sleeping_mask=sleeping)
    return sleeping, counters


def _set_sleep_state(
    *, context: dict[str, Any], sleeping: torch.Tensor, counters: torch.Tensor, sleeping_mask: torch.Tensor
) -> None:
    sleep_ctx = context.setdefault("sleep", {})
    sleep_ctx["state"] = {"sleeping": sleeping, "counters": counters}
    sleep_ctx["sleeping_mask"] = sleeping_mask


def _ensure_sleep_context(
    *, context: dict[str, Any], state: State, sleeping_mask: torch.Tensor | None, freeze_core: bool
) -> None:
    sleep_ctx = context.setdefault("sleep", {})
    sleep_ctx["freeze_core"] = bool(freeze_core)
    if sleeping_mask is not None:
        sleep_ctx["sleeping_mask"] = sleeping_mask
    sleep_ctx.setdefault("last_t", state.t)


def _build_islands(contacts: list[dict[str, Any]], *, n_bodies: int) -> list[list[int]]:
    parent = list(range(n_bodies))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for c in contacts:
        try:
            mode = int(c.get("mode", 0))
            if mode in {int(ContactMode.SEPARATED), int(ContactMode.IMPACT)}:
                continue
            body_i = int(c.get("body_i", -1))
            body_j = int(c.get("body_j", -1))
        except Exception:
            continue

        if body_i >= 0 and body_j >= 0 and body_i < n_bodies and body_j < n_bodies:
            union(body_i, body_j)

    groups: dict[int, list[int]] = {}
    for i in range(n_bodies):
        root = find(i)
        groups.setdefault(root, []).append(i)
    return list(groups.values())


def _supported_by_static(contacts: list[dict[str, Any]], *, n_bodies: int) -> torch.Tensor:
    supported = torch.zeros((n_bodies,), dtype=torch.bool)
    for c in contacts:
        try:
            mode = int(c.get("mode", 0))
            if mode in {int(ContactMode.SEPARATED), int(ContactMode.IMPACT)}:
                continue
            body_i = int(c.get("body_i", -1))
            body_j = int(c.get("body_j", -1))
        except Exception:
            continue

        if body_i >= 0 and body_i < n_bodies and body_j < 0:
            supported[body_i] = True
        if body_j >= 0 and body_j < n_bodies and body_i < 0:
            supported[body_j] = True
    return supported


def _bodies_with_mode(contacts: list[dict[str, Any]], *, mode: int, n_bodies: int) -> torch.Tensor:
    mask = torch.zeros((n_bodies,), dtype=torch.bool)
    for c in contacts:
        try:
            if int(c.get("mode", 0)) != int(mode):
                continue
            body_i = int(c.get("body_i", -1))
            body_j = int(c.get("body_j", -1))
        except Exception:
            continue
        if 0 <= body_i < n_bodies:
            mask[body_i] = True
        if 0 <= body_j < n_bodies:
            mask[body_j] = True
    return mask


def _apply_sleep_to_state(state: State, sleeping_mask: torch.Tensor) -> State:
    if not bool(sleeping_mask.any().item()):
        return state

    mask = sleeping_mask.to(device=state.v.device)
    while mask.ndim < state.v.ndim:
        mask = mask.unsqueeze(-1)

    next_v = torch.where(mask, torch.zeros_like(state.v), state.v)
    return State(q=state.q, v=next_v, t=state.t)


def _apply_sleep_to_contacts(contacts: list[dict[str, Any]], sleeping_mask: torch.Tensor) -> list[dict[str, Any]]:
    if not bool(sleeping_mask.any().item()):
        return contacts

    sleeping_indices = set(int(i) for i in torch.nonzero(sleeping_mask).flatten().tolist())
    next_contacts: list[dict[str, Any]] = []
    for c in contacts:
        body_i = int(c.get("body_i", -1))
        body_j = int(c.get("body_j", -1))

        i_sleep = body_i in sleeping_indices
        j_sleep = body_j in sleeping_indices
        if (i_sleep and body_j < 0) or (j_sleep and body_i < 0) or (i_sleep and j_sleep):
            c = dict(c)
            c["mode"] = int(ContactMode.RESTING)
            c["mode_name"] = contact_mode_name(int(ContactMode.RESTING))
        next_contacts.append(c)
    return next_contacts
