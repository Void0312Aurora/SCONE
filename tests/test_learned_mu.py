import torch

from scone.learned.mu import ContactMuMLP, ContactMuMaterialMLP, ScalarMu
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


def test_contact_mu_mlp_output_is_bounded_and_differentiable() -> None:
    device = torch.device("cpu")
    dtype = torch.float32

    model = ContactMuMLP(init_mu=0.3, mu_max=1.0, hidden_dim=8, device=device, dtype=dtype)
    mu = model(
        phi=torch.tensor(-0.01, device=device, dtype=dtype),
        vn=torch.tensor(-1.5, device=device, dtype=dtype),
        vt=torch.tensor(2.0, device=device, dtype=dtype),
        is_pair=torch.tensor(1.0, device=device, dtype=dtype),
    )
    assert float(mu.detach().cpu().item()) >= 0.0
    assert float(mu.detach().cpu().item()) <= 1.0

    target = torch.tensor(0.7, device=device, dtype=dtype)
    loss = (mu - target) ** 2
    loss.backward()

    total_grad = torch.tensor(0.0, device=device, dtype=dtype)
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
            total_grad = total_grad + param.grad.abs().sum()
    assert float(total_grad.detach().cpu().item()) > 0.0


def test_contact_mu_material_mlp_output_is_bounded_and_differentiable() -> None:
    device = torch.device("cpu")
    dtype = torch.float32

    model = ContactMuMaterialMLP(
        init_mu=0.3,
        mu_max=1.0,
        hidden_dim=8,
        max_material_id=4,
        device=device,
        dtype=dtype,
    )
    mu = model(
        phi=torch.tensor(-0.01, device=device, dtype=dtype),
        vn=torch.tensor(-1.5, device=device, dtype=dtype),
        vt=torch.tensor(2.0, device=device, dtype=dtype),
        is_pair=torch.tensor(1.0, device=device, dtype=dtype),
        material_i=torch.tensor(2.0, device=device, dtype=dtype),
        material_j=torch.tensor(4.0, device=device, dtype=dtype),
    )
    assert float(mu.detach().cpu().item()) >= 0.0
    assert float(mu.detach().cpu().item()) <= 1.0

    target = torch.tensor(0.7, device=device, dtype=dtype)
    loss = (mu - target) ** 2
    loss.backward()

    total_grad = torch.tensor(0.0, device=device, dtype=dtype)
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
            total_grad = total_grad + param.grad.abs().sum()
    assert float(total_grad.detach().cpu().item()) > 0.0
