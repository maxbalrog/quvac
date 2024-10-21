'''
Here we provide a test for quantum vacuum signal calculator
with script
'''

import os
from pathlib import Path

import pytest
import numpy as np
from scipy.constants import c

from quvac.grid import get_xyz_size, get_t_size
from quvac.utils import write_yaml


SCRIPT_PATH = 'src/quvac/simulation.py'


def test_spherical_grid():
    # Define field parameters
    tau = 25e-15
    W = 25
    lam = 0.8e-6
    w0 = 2*lam
    theta = 180
    beta = 90

    # Define fields
    field_1 = {
        "field_type": "paraxial_gaussian_analytic",
        "focus_x": [0.,0.,0.],
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
        "focus_x": [0.,0.,0.],
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

    # Set up grid parameters
    x0, y0, z0 = 15*w0, 15*w0, 6*c*tau
    box_size = np.array([x0, y0, z0])/2
    Nxyz = get_xyz_size(fields_params, box_size)
    Nx, Ny, Nz = Nxyz

    t0 = 4*tau
    Nt = get_t_size(-t0/2, t0/2, lam)

    ini_data = {
        'fields': fields_params,
        'grid': {
            'mode': 'direct',
            'box_xyz': [x0,y0,z0],
            'Nxyz': [Nx,Ny,Nz],
            'box_t': t0,
            'Nt': Nt
        },
        'performance': {},
        'postprocess': {
            'calculate_spherical': True,
            'calculate_discernible': False,
        },
    }

    path = 'data/test/test_sph'
    Path(path).mkdir(parents=True, exist_ok=True)

    ini_file = os.path.join(path, 'ini.yml')
    write_yaml(ini_file, ini_data)

    # Launch simulation
    status = os.system(f"{SCRIPT_PATH} --input {ini_file}")
    assert status == 0, "Script execution did not finish successfully"

    data = np.load(os.path.join(path, 'spectra.npz'))
    err_msg = 'Total signal on cartesian and spherical grid differ by more than 1%'
    assert np.isclose(data['N_total'], data['N_sph_total'], rtol=1e-2), err_msg


@pytest.mark.slow
def test_discernible():
    # Define field parameters
    tau = 25e-15
    W = 25
    lam = 0.8e-6
    w01 = 1*lam
    w02 = 4*lam
    theta = 160
    beta = 90

    # Define fields
    field_1 = {
        "field_type": "paraxial_gaussian_maxwell",
        "focus_x": [0.,0.,0.],
        "focus_t": 0.,
        "theta": 0,
        "phi": 0,
        "beta": 0,
        "lam": lam,
        "w0": w01,
        "tau": tau,
        "W": W,
        "phase0": 0,
    }

    field_2 = {
        "field_type": "paraxial_gaussian_maxwell",
        "focus_x": [0.,0.,0.],
        "focus_t": 0.,
        "theta": theta,
        "phi": 0,
        "beta": beta,
        "lam": lam,
        "w0": w02,
        "tau": tau,
        "W": W,
        "phase0": 0,
    }

    fields_params = [field_1, field_2]

    # Set up grid parameters

    ini_data = {
        'fields': fields_params,
        'grid': {
            'mode': 'dynamic',
            'collision_geometry': 'z',
            'transverse_factor': 15,
            'longitudinal_factor': 6,
            'time_factor': 4,
            'spatial_resolution': 1,
            'time_resolution': 1,
        },
        'performance': {
            'nthreads': 8,
        },
        'postprocess': {
            'calculate_spherical': True,
            'calculate_discernible': True,
        },
    }


    path = 'data/test/test_sph'
    Path(path).mkdir(parents=True, exist_ok=True)

    ini_file = os.path.join(path, 'ini.yml')
    write_yaml(ini_file, ini_data)

    # Launch simulation
    status = os.system(f"{SCRIPT_PATH} --input {ini_file}")
    assert status == 0, "Script execution did not finish successfully"

    data = np.load(os.path.join(path, 'spectra.npz'))
    N_disc = data['N_disc']
    N_disc_expexted = 2.5
    err_msg = 'Calculated discernible signal differs from expected by more than 20%'
    assert np.isclose(N_disc, N_disc_expexted, rtol=2e-1), err_msg
