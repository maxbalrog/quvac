'''
Here we provide a test for vacuum emission integrator:
    - Given two colliding paraxial gaussians we calculate the total
    vacuum signal and compare it with analytic calculation
'''

import numpy as np
from scipy.constants import pi, c

from quvac.field.paraxial_gaussian import ParaxialGaussianAnalytic
from quvac.field.external_field import ExternalField
from quvac.integrator.vacuum_emission import VacuumEmission
from quvac.grid_utils import get_xyz_size, get_t_size
from quvac.analytic_scalings import get_two_paraxial_scaling


def test_two_paraxial_gaussians():
    # Define field parameters
    tau = 25e-15
    W = 25
    lam = 0.8e-6
    w0 = 2*lam
    theta = 90
    beta = 90

    # Define fields
    field_1 = {
        "field_type": "paraxial_gaussian_analytic",
        "focus_x": (0.,0.,0.),
        "focus_t": 0.,
        "theta": 0,
        "phi": 0,
        "beta": 0,
        "lam": lam,
        "w0": w0,
        "tau": tau,
        "W": W,
        "phase0": 0,
    }

    field_2 = {
        "field_type": "paraxial_gaussian_analytic",
        "focus_x": (0.,0.,0.),
        "focus_t": 0.,
        "theta": theta,
        "phi": 0,
        "beta": beta,
        "lam": lam,
        "w0": w0,
        "tau": tau,
        "W": W,
        "phase0": 0,
    }

    fields_params = [field_1, field_2]

    x0, y0, z0 = 5*c*tau, 12*w0, 5*c*tau
    box_size = np.array([x0, y0, z0])
    Nxyz = get_xyz_size(fields_params, box_size/2)
    Nx, Ny, Nz = Nxyz
    x = np.linspace(-x0/2,x0/2,Nx).reshape((-1,1,1))
    y = np.linspace(-y0/2,y0/2,Ny).reshape((1,-1,1))
    z = np.linspace(-z0/2,z0/2,Nz).reshape((1,1,-1))
    grid = (x, y, z)

    t0 = 2*tau
    Nt = get_t_size(-t0, t0, lam)
    t_grid = np.linspace(-t0, t0, Nt)

    # Calculate analytic scalings
    N_signal_th, N_perp_th = get_two_paraxial_scaling(fields_params)

    # Calculate signal numerically
    field = ExternalField(fields_params, grid)
    vacem = VacuumEmission(field)
    vacem.calculate_vacuum_current(t_grid)
    N_signal_num = vacem.calculate_total_signal()

    print(f"Total signal (theory): {N_signal_th:.3f}")
    print(f"Total signal (num)   : {N_signal_num:.3f}")

    err_msg = "Analytical and numerical total signal differ by more than 10%"
    assert np.isclose(N_signal_th, N_signal_num, rtol=1e-1), err_msg