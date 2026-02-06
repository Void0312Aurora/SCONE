import torch

from scone.layers.constraints import LinearConstraintProjectionLayer
from scone.state import State


def test_linear_constraint_projection_projects_position_and_velocity() -> None:
    layer = LinearConstraintProjectionLayer(eps=1e-9, enforce_position=True, enforce_velocity=True)
    state = State(
        q=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        v=torch.tensor([[3.0, 4.0]], dtype=torch.float64),
        t=0.0,
    )
    context = {
        "constraints": {
            "A_pos": torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            "b_pos": torch.tensor([0.5], dtype=torch.float64),
            "A_vel": torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            "b_vel": torch.tensor([0.0], dtype=torch.float64),
        }
    }

    next_state, diag = layer.project(state=state, dt=0.01, context=context)

    assert abs(float(next_state.q[0, 0].item()) - 0.5) < 1e-8
    assert abs(float(next_state.q[0, 1].item()) - 2.0) < 1e-8
    assert abs(float(next_state.v[0, 0].item()) - 3.0) < 1e-8
    assert abs(float(next_state.v[0, 1].item()) - 0.0) < 1e-8

    constraints = diag["constraints"]
    assert float(constraints["residual_pos"].item()) < 1e-8
    assert float(constraints["residual_vel"].item()) < 1e-8
    assert float(constraints["power_error"].item()) < 1e-5


def test_linear_constraint_projection_no_constraints_is_noop() -> None:
    layer = LinearConstraintProjectionLayer()
    state = State(
        q=torch.tensor([[1.2, -0.3]], dtype=torch.float64),
        v=torch.tensor([[0.8, 2.1]], dtype=torch.float64),
        t=0.0,
    )

    next_state, diag = layer.project(state=state, dt=0.01, context={})

    assert torch.allclose(next_state.q, state.q)
    assert torch.allclose(next_state.v, state.v)
    assert float(diag["constraints"]["residual_pos"].item()) == 0.0
    assert float(diag["constraints"]["residual_vel"].item()) == 0.0
    assert float(diag["constraints"]["power_error"].item()) == 0.0
