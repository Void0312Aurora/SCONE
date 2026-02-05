from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class ContactMode(IntEnum):
    SEPARATED = 0
    IMPACT = 1
    RESTING = 2
    ACTIVE = 3
    SLIDING = 4
    STICKING = 5


def contact_mode_name(mode: int) -> str:
    try:
        return ContactMode(int(mode)).name.lower()
    except ValueError:
        return "unknown"


@dataclass(frozen=True)
class Contact:
    id: str
    body_i: int
    body_j: int
    phi: torch.Tensor
    n: torch.Tensor
    lambda_n: torch.Tensor
    lambda_t: torch.Tensor
    mode: int

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "body_i": int(self.body_i),
            "body_j": int(self.body_j),
            "phi": self.phi,
            "n": self.n,
            "lambda_n": self.lambda_n,
            "lambda_t": self.lambda_t,
            "mode": int(self.mode),
            "mode_name": contact_mode_name(int(self.mode)),
        }

