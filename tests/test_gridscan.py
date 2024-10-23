'''
Here we provide a test for quantum vacuum signal calculator
with script
'''

import os
from pathlib import Path

import numpy as np
from scipy.constants import c

from quvac.grid import get_xyz_size, get_t_size
from quvac.utils import write_yaml


SCRIPT_PATH = 'src/quvac/cluster/gridscan.py'


def test_gridscan():
    # Define field parameters
    tau = 25e-15
    W = 25
    lam = 0.8e-6
    w01 = 1*lam
    w02 = 1*lam
    theta = 180
    beta = 90

    mode = 'maxwell'

    # Define fields
    field_1 = {
        "field_type": f"paraxial_gaussian_{mode}",
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
        "field_type": f"paraxial_gaussian_{mode}",
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

    fields_params = {
        'field_1': field_1,
        'field_2': field_2,
    }

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
            'nthreads': 8
        },
        'postprocess': {
            'calculate_spherical': False,
            'calculate_discernible': False,
        },
    }

    beta_arr = [0, 45, 90]
    variables_data = {
        'create_grids': True,
        'fields': {
            'field_2': {
                'beta': [0, 90, 3]
            }
        },
        'cluster': {
            'cluster': 'local',
            'max_parallel_jobs': 10,
        }
    }

    path = 'data/test/test_gridscan'
    Path(path).mkdir(parents=True, exist_ok=True)

    ini_file = os.path.join(path, 'ini.yaml')
    write_yaml(ini_file, ini_data)

    variables_file = os.path.join(path, 'variables.yaml')
    write_yaml(variables_file, variables_data)

    # Launch simulation
    status = os.system(f"{SCRIPT_PATH} --input {ini_file} --variables {variables_file}")
    assert status == 0, "Script execution did not finish successfully"

    folders = [f'#field_2:beta_{beta}' for beta in beta_arr]
    data = []
    for folder in folders:
        data_loc = np.load(os.path.join(path, folder, 'spectra.npz'))
        data.append(data_loc['N_total'])
    
    err_msg = 'Calculated results do not agree with analytics'
    assert np.isclose(data[1]/data[0], 130/64, rtol=1e-1), err_msg


