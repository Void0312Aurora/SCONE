from __future__ import annotations

import torch


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
        p = init_clamped / float(mu_max)
        p = float(max(1e-6, min(1.0 - 1e-6, p)))
        raw = torch.logit(torch.tensor(p, device=device, dtype=dtype))
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
            p = init_clamped / float(mu_max)
            p = float(max(1e-6, min(1.0 - 1e-6, p)))
            return torch.logit(torch.tensor(p, device=device, dtype=dtype))

        self.mu_max = float(mu_max)
        self.raw_ground = torch.nn.Parameter(raw_from_init(float(init_ground)))
        self.raw_pair = torch.nn.Parameter(raw_from_init(float(init_pair)))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        mu_ground = torch.sigmoid(self.raw_ground) * self.mu_max
        mu_pair = torch.sigmoid(self.raw_pair) * self.mu_max
        return mu_ground, mu_pair
