import torch

from scone.engine import Engine
from scone.layers.constraints import NoOpConstraintLayer
from scone.layers.dissipation import LinearDampingLayer
from scone.layers.events import DiskContactPGSEventLayer
from scone.layers.symplectic import SymplecticEulerSeparable
from scone.state import State
from scone.systems.rigid2d import Disk2D


def test_disk_stack_pgs_multi_contact_supports_top_disk() -> None:
    device = torch.device("cpu")
    dtype = torch.float64

    mass = 1.0
    radius = 0.5
    inertia = 0.5 * mass * radius * radius
    gravity = 9.81

    system = Disk2D(
        mass=mass,
        radius=radius,
        gravity=gravity,
        ground_height=0.0,
        inertia=inertia,
        device=device,
        dtype=dtype,
    )
    engine = Engine(
        system=system,
        core=SymplecticEulerSeparable(system=system),
        dissipation=LinearDampingLayer(damping=0.0),
        constraints=NoOpConstraintLayer(),
        events=DiskContactPGSEventLayer(
            mass=mass,
            inertia=inertia,
            radius=radius,
            gravity=gravity,
            friction_mu=0.6,
            restitution=0.0,
            ground_height=0.0,
            contact_slop=1e-3,
            impact_velocity_min=0.2,
            pgs_iters=30,
            baumgarte_beta=0.2,
            sleep=None,
        ),
    )

    state = State(
        q=torch.tensor(
            [
                [-0.6, 0.5, 0.0],
                [0.6, 0.5, 0.0],
                [0.0, 1.3, 0.0],
            ],
            device=device,
            dtype=dtype,
        ),
        v=torch.zeros((3, 3), device=device, dtype=dtype),
        t=0.0,
    )

    next_state, diag = engine.step(state=state, dt=0.01, context={})

    assert diag["failsafe"]["triggered"] is False

    contacts = diag["contacts"]["items"]
    assert isinstance(contacts, list)
    assert len(contacts) == 4
    ids = {c["id"] for c in contacts}
    assert ids == {"disk0-ground", "disk1-ground", "disk0-disk2", "disk1-disk2"}

    for c in contacts:
        for key in ["id", "body_i", "body_j", "phi", "n", "lambda_n", "lambda_t", "mode", "mode_name", "phi_next"]:
            assert key in c

    # Bottom disks are supported by ground; top disk is supported by two contacts.
    assert abs(float(next_state.v[0, 1].item())) < 1e-6
    assert abs(float(next_state.v[1, 1].item())) < 1e-6
    assert abs(float(next_state.v[2, 1].item())) < 1e-6

    # Penetration stays within a small tolerance (on the order of g*dt^2).
    assert float(diag["contacts"]["penetration_max"].item()) < 2e-3
