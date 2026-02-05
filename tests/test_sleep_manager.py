import torch

from scone.contacts import ContactMode
from scone.sleep import SleepConfig, SleepManager
from scone.state import State


def test_sleep_manager_sleeps_supported_island() -> None:
    manager = SleepManager(
        SleepConfig(enabled=True, v_sleep=0.1, v_wake=0.2, steps_to_sleep=1, freeze_core=True)
    )
    state = State(
        q=torch.zeros((1, 3), dtype=torch.float64),
        v=torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64),
        t=0.0,
    )
    contacts = [{"id": "body0-ground", "body_i": 0, "body_j": -1, "mode": int(ContactMode.ACTIVE)}]
    context: dict[str, object] = {}

    next_state, next_contacts, diag = manager.apply(state=state, contacts=contacts, context=context)

    assert torch.allclose(next_state.v, torch.zeros_like(next_state.v))
    assert "sleep" in diag
    sleeping_mask = context["sleep"]["sleeping_mask"]  # type: ignore[index]
    assert isinstance(sleeping_mask, torch.Tensor)
    assert bool(sleeping_mask[0].item()) is True
    assert int(next_contacts[0]["mode"]) == int(ContactMode.RESTING)

