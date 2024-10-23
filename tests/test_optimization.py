'''
Here we provide a test for oftimization of quvac
simulations
'''

import os
from pathlib import Path

import pytest
import numpy as np
from scipy.constants import c
from ax.service.ax_client import AxClient

from quvac.utils import write_yaml
from quvac.cluster.optimization import gather_trials_data


SCRIPT_PATH = 'src/quvac/cluster/optimization.py'

@pytest.mark.slow
def test_optimization():
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

    path = 'data/test/test_optimization'
    Path(path).mkdir(parents=True, exist_ok=True)

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
            'calculate_spherical': True,
            'calculate_discernible': True,
        },
        'scales': {
            
        },
        'save_path': path
    }

    optimization_data = {
        'name': '2_pulses_beta',
        'parameters': {
            'field_2': {
                'beta': [0, 90]
            }
        },
        'cluster': {
            'cluster': 'local',
        },
        'n_trials': 10,
        'objectives': [['N_total', False]],
    }

    ini_file = os.path.join(path, 'ini.yml')
    write_yaml(ini_file, ini_data)

    optimization_file = os.path.join(path, 'optimization.yml')
    write_yaml(optimization_file, optimization_data)

    # Launch simulation
    status = os.system(f"{SCRIPT_PATH} --input {ini_file} --optimization {optimization_file}")
    assert status == 0, "Script execution did not finish successfully"

    client_json = os.path.join(path, 'experiment.json')
    ax_client = (AxClient.load_from_json_file(client_json))

    trials_params = gather_trials_data(ax_client, metric_names=['N_total'])
    betas = [val['field_2:beta'] for val in trials_params.values()]
    N_total = [val['N_total'] for val in trials_params.values()]
    beta_max = betas[np.argmax(N_total)]
    
    err_msg = 'Optimization-found value of beta is not close to the real optimum'
    assert np.isclose(beta_max, 90., rtol=1e-1), err_msg