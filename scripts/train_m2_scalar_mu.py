from __future__ import annotations

import argparse

import torch

from scone.learned.mu import ScalarMu
from scone.layers.events import DiskGroundContactEventLayer
from scone.state import State


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--mu-true", type=float, default=0.6)
    parser.add_argument("--mu-init", type=float, default=0.1)
    parser.add_argument("--mu-max", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cpu")
    dtype = torch.float64

    # A single-step supervised signal: pick a state that is penetrating and moving down so that
    # the normal impulse is active and friction depends on mu.
    state = State(
        q=torch.tensor([[0.0, 0.499019, 0.0]], device=device, dtype=dtype),
        v=torch.tensor([[2.0, -0.0981, 0.0]], device=device, dtype=dtype),
        t=0.0,
    )

    target_layer = DiskGroundContactEventLayer(
        mass=1.0,
        inertia=0.125,
        radius=0.5,
        gravity=9.81,
        friction_mu=float(args.mu_true),
        restitution=0.0,
        ground_height=0.0,
        contact_slop=1e-3,
        impact_velocity_min=0.2,
        sleep=None,
    )
    target_state, _ = target_layer.resolve(state=state, dt=0.01, context={})
    target_v = target_state.v.detach()

    model = ScalarMu(init=float(args.mu_init), mu_max=float(args.mu_max), device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)
        mu = model()
        layer = DiskGroundContactEventLayer(
            mass=1.0,
            inertia=0.125,
            radius=0.5,
            gravity=9.81,
            friction_mu=mu,
            restitution=0.0,
            ground_height=0.0,
            contact_slop=1e-3,
            impact_velocity_min=0.2,
            sleep=None,
        )
        pred_state, _ = layer.resolve(state=state, dt=0.01, context={})
        loss = torch.nn.functional.mse_loss(pred_state.v, target_v)
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == int(args.steps) - 1:
            print(f"step={step:03d} loss={float(loss.detach()):.6e} mu={float(model().detach()):.6f}")

    print(f"done: mu_true={float(args.mu_true):.6f} mu_fit={float(model().detach()):.6f}")


if __name__ == "__main__":
    main()

