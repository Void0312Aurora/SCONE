from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class State:
    q: torch.Tensor
    v: torch.Tensor
    t: float

    def as_dict(self) -> dict[str, object]:
        return {"q": self.q, "v": self.v, "t": self.t}

