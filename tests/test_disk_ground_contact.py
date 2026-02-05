import torch

from scone.layers.events import DiskGroundContactEventLayer
from scone.state import State


def test_disk_ground_contact_generates_contact_and_rolls() -> None:
    layer = DiskGroundContactEventLayer(
        mass=1.0,
        inertia=0.125,  # 0.5 * m * r^2 for r=0.5
        radius=0.5,
        gravity=9.81,
        friction_mu=0.6,
        restitution=0.0,
        ground_height=0.0,
        contact_slop=1e-3,
        impact_velocity_min=0.2,
        sleep=None,
    )

    state = State(
        q=torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float64),
        v=torch.tensor([[1.0, -0.1, 0.0]], dtype=torch.float64),
        t=0.0,
    )

    next_state, diag = layer.resolve(state=state, dt=0.01, context={})

    items = diag["contacts"]["items"]
    assert isinstance(items, list)
    assert len(items) == 1
    c = items[0]
    for key in ["id", "body_i", "body_j", "phi", "n", "lambda_n", "lambda_t", "mode", "mode_name", "phi_next"]:
        assert key in c

    assert float(c["lambda_n"]) > 0.0
    assert isinstance(c["lambda_t"], list) or isinstance(c["lambda_t"], torch.Tensor)

    # Friction should reduce vx magnitude and induce rotation (omega != 0).
    assert float(next_state.v[0, 0].item()) < float(state.v[0, 0].item())
    assert abs(float(next_state.v[0, 2].item())) > 0.0

