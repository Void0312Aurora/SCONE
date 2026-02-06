import torch

from scone.learned.mu import ScalarMu
from scone.layers.events import DiskGroundContactEventLayer
from scone.state import State


def test_scalar_mu_can_be_fit_from_one_supervised_step() -> None:
    device = torch.device("cpu")
    dtype = torch.float64

    state = State(
        q=torch.tensor([[0.0, 0.499019, 0.0]], device=device, dtype=dtype),
        v=torch.tensor([[2.0, -0.0981, 0.0]], device=device, dtype=dtype),
        t=0.0,
    )

    mu_true = 0.6
    target_layer = DiskGroundContactEventLayer(
        mass=1.0,
        inertia=0.125,
        radius=0.5,
        gravity=9.81,
        friction_mu=mu_true,
        restitution=0.0,
        ground_height=0.0,
        contact_slop=1e-3,
        impact_velocity_min=0.2,
        sleep=None,
    )
    target_state, _ = target_layer.resolve(state=state, dt=0.01, context={})
    target_v = target_state.v.detach()

    model = ScalarMu(init=0.1, mu_max=1.0, device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=0.5)

    loss0 = None
    for _ in range(40):
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
        if loss0 is None:
            loss0 = float(loss.detach().cpu().item())
        loss.backward()
        opt.step()

    mu_fit = float(model().detach().cpu().item())
    assert abs(mu_fit - mu_true) < 5e-2
    assert float(loss.detach().cpu().item()) < 0.1 * float(loss0)

