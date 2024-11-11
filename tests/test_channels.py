'''
Test channel separation in the calculation of vacuum emission
amplitude
'''

import os
from pathlib import Path

import numpy as np
from scipy.constants import c

from quvac.grid import get_xyz_size, get_t_size
from quvac.utils import write_yaml


SCRIPT_PATH = 'src/quvac/simulation.py'


def test_channels():
    # Define field parameters
    tau = 25e-15
    W = 25
    lam = 0.8e-6
    w0 = 1*lam
    theta = 180
    beta = 0

    mode = 'analytic'
    order = 0

    # Define fields
    field_1 = {
        "field_type": f"paraxial_gaussian_{mode}",
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
        "order": order,
    }

    field_2 = {
        "field_type": f"paraxial_gaussian_{mode}",
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
        "order": order,
    }

    fields_params = {
        'field_1': field_1,
        'field_2': field_2,
    }

    results = []
    for idx,channels in zip([(0,0), (0,1), (1,0)], [False, True, True]):
        integrator_type = 'vacuum_emission_channels' if channels else 'vacuum_emission'

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
            'integrator': {
                'type': integrator_type,
                'probe_pump_idx': {
                    'probe': [idx[0]],
                    'pump': [idx[1]],
                }
            },
        }

        path = 'data/test/test_sim'
        Path(path).mkdir(parents=True, exist_ok=True)

        ini_file = os.path.join(path, 'ini.yml')
        write_yaml(ini_file, ini_data)

        # Launch simulation
        status = os.system(f"{SCRIPT_PATH} --input {ini_file}")
        assert status == 0, "Script execution did not finish successfully"

        data_file = os.path.join(path, 'spectra.npz')
        data = np.load(data_file)
        
        N_signal_num = data['N_total']
        results.append(N_signal_num)
    
    # Test that signal_channel_1 + signal_channel_2 = total_signal
    # and signal_channel_1 == signal_channel_2
    err_msg = 'Signal is different in two channels'
    assert np.isclose(results[1], results[2], rtol=1e-5), err_msg
    err_msg = 'Signal in two channels does not add up to the total signal'
    assert np.isclose(results[0], results[1] + results[2], rtol=1e-5), err_msg

