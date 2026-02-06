from __future__ import annotations

import torch


def _clamped_logit_from_ratio(ratio: float, *, device: torch.device | None, dtype: torch.dtype | None) -> torch.Tensor:
    p = float(max(1e-6, min(1.0 - 1e-6, ratio)))
    return torch.logit(torch.tensor(p, device=device, dtype=dtype))


def _as_scalar(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 0:
        return x
    return x.reshape(-1)[0]


class ScalarMu(torch.nn.Module):
    def __init__(
        self,
        *,
        init: float = 0.5,
        mu_max: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if mu_max <= 0.0:
            raise ValueError("mu_max must be positive")

        init_clamped = float(max(0.0, min(init, mu_max)))
        raw = _clamped_logit_from_ratio(init_clamped / float(mu_max), device=device, dtype=dtype)
        self.mu_max = float(mu_max)
        self.raw = torch.nn.Parameter(raw)

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw) * self.mu_max


class GroundPairMu(torch.nn.Module):
    def __init__(
        self,
        *,
        init_ground: float = 0.5,
        init_pair: float | None = None,
        mu_max: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if mu_max <= 0.0:
            raise ValueError("mu_max must be positive")

        if init_pair is None:
            init_pair = float(init_ground)

        def raw_from_init(init: float) -> torch.Tensor:
            init_clamped = float(max(0.0, min(init, mu_max)))
            return _clamped_logit_from_ratio(init_clamped / float(mu_max), device=device, dtype=dtype)

        self.mu_max = float(mu_max)
        self.raw_ground = torch.nn.Parameter(raw_from_init(float(init_ground)))
        self.raw_pair = torch.nn.Parameter(raw_from_init(float(init_pair)))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        mu_ground = torch.sigmoid(self.raw_ground) * self.mu_max
        mu_pair = torch.sigmoid(self.raw_pair) * self.mu_max
        return mu_ground, mu_pair


class ContactMuMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        init_mu: float = 0.4,
        mu_max: float = 1.0,
        hidden_dim: int = 16,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if mu_max <= 0.0:
            raise ValueError("mu_max must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.mu_max = float(mu_max)
        self.vn_scale = 3.0
        self.vt_scale = 3.0

        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, int(hidden_dim), device=device, dtype=dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(int(hidden_dim), 1, device=device, dtype=dtype),
        )
        self._reset(init_mu=float(init_mu))

    def _reset(self, *, init_mu: float) -> None:
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        init_clamped = float(max(0.0, min(init_mu, self.mu_max)))
        raw_bias = _clamped_logit_from_ratio(
            init_clamped / float(self.mu_max),
            device=self.net[-1].bias.device,
            dtype=self.net[-1].bias.dtype,
        )
        with torch.no_grad():
            last = self.net[-1]
            assert isinstance(last, torch.nn.Linear)
            last.weight.mul_(0.1)
            last.bias.copy_(raw_bias)

    def _encode_features(
        self,
        *,
        phi: torch.Tensor,
        vn: torch.Tensor,
        vt: torch.Tensor,
        is_pair: torch.Tensor,
    ) -> torch.Tensor:
        phi_s = _as_scalar(phi)
        vn_s = _as_scalar(vn)
        vt_s = _as_scalar(vt)
        is_pair_s = _as_scalar(is_pair)

        penetration = torch.clamp(-phi_s, min=0.0, max=1.0)
        vn_abs = torch.clamp(vn_s.abs() / self.vn_scale, min=0.0, max=4.0)
        vt_abs = torch.clamp(vt_s.abs() / self.vt_scale, min=0.0, max=4.0)
        pair_flag = torch.clamp(is_pair_s, min=0.0, max=1.0)
        return torch.stack([penetration, vn_abs, vt_abs, pair_flag], dim=0)

    def forward(
        self,
        *,
        phi: torch.Tensor,
        vn: torch.Tensor,
        vt: torch.Tensor,
        is_pair: torch.Tensor,
    ) -> torch.Tensor:
        x = self._encode_features(phi=phi, vn=vn, vt=vt, is_pair=is_pair).unsqueeze(0)
        raw = self.net(x).reshape(())
        return torch.sigmoid(raw) * self.mu_max

    def from_contact_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self(
            phi=features["phi"],
            vn=features["vn"],
            vt=features["vt"],
            is_pair=features["is_pair"],
        )


class ContactMuMaterialMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        init_mu: float = 0.4,
        mu_max: float = 1.0,
        hidden_dim: int = 24,
        max_material_id: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if mu_max <= 0.0:
            raise ValueError("mu_max must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if max_material_id <= 0:
            raise ValueError("max_material_id must be positive")

        self.mu_max = float(mu_max)
        self.vn_scale = 3.0
        self.vt_scale = 3.0
        self.max_material_id = float(max_material_id)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, int(hidden_dim), device=device, dtype=dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim), device=device, dtype=dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(int(hidden_dim), 1, device=device, dtype=dtype),
        )
        self._reset(init_mu=float(init_mu))

    def _reset(self, *, init_mu: float) -> None:
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        init_clamped = float(max(0.0, min(init_mu, self.mu_max)))
        raw_bias = _clamped_logit_from_ratio(
            init_clamped / float(self.mu_max),
            device=self.net[-1].bias.device,
            dtype=self.net[-1].bias.dtype,
        )
        with torch.no_grad():
            last = self.net[-1]
            assert isinstance(last, torch.nn.Linear)
            last.weight.mul_(0.1)
            last.bias.copy_(raw_bias)

    def _encode_features(
        self,
        *,
        phi: torch.Tensor,
        vn: torch.Tensor,
        vt: torch.Tensor,
        is_pair: torch.Tensor,
        material_i: torch.Tensor,
        material_j: torch.Tensor,
    ) -> torch.Tensor:
        phi_s = _as_scalar(phi)
        vn_s = _as_scalar(vn)
        vt_s = _as_scalar(vt)
        is_pair_s = _as_scalar(is_pair)
        mat_i_s = _as_scalar(material_i)
        mat_j_s = _as_scalar(material_j)

        penetration = torch.clamp(-phi_s, min=0.0, max=1.0)
        vn_abs = torch.clamp(vn_s.abs() / self.vn_scale, min=0.0, max=4.0)
        vt_abs = torch.clamp(vt_s.abs() / self.vt_scale, min=0.0, max=4.0)
        pair_flag = torch.clamp(is_pair_s, min=0.0, max=1.0)
        mat_i = torch.clamp(mat_i_s / self.max_material_id, min=0.0, max=1.0)
        mat_j = torch.clamp(mat_j_s / self.max_material_id, min=0.0, max=1.0)
        return torch.stack([penetration, vn_abs, vt_abs, pair_flag, mat_i, mat_j], dim=0)

    def forward(
        self,
        *,
        phi: torch.Tensor,
        vn: torch.Tensor,
        vt: torch.Tensor,
        is_pair: torch.Tensor,
        material_i: torch.Tensor,
        material_j: torch.Tensor,
    ) -> torch.Tensor:
        x = self._encode_features(
            phi=phi,
            vn=vn,
            vt=vt,
            is_pair=is_pair,
            material_i=material_i,
            material_j=material_j,
        ).unsqueeze(0)
        raw = self.net(x).reshape(())
        return torch.sigmoid(raw) * self.mu_max

    def from_contact_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        reference = features["phi"]
        zero = torch.tensor(0.0, device=reference.device, dtype=reference.dtype)
        return self(
            phi=features["phi"],
            vn=features["vn"],
            vt=features["vt"],
            is_pair=features["is_pair"],
            material_i=features.get("material_i", zero),
            material_j=features.get("material_j", zero),
        )
