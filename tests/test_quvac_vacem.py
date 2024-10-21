'''
Here we provide a comparison between quvac and vacem results
'''

import os
from pathlib import Path

import pytest
import numpy as np
from scipy.constants import c

from quvac.grid import get_xyz_size, get_t_size
from quvac.utils import write_yaml, read_yaml


SCRIPT_PATH = 'src/quvac/simulation.py'
REFERENCE_PATH = 'tests/references'
REFERENCE_FILE = 'results.yml'


@pytest.mark.slow
def test_theta_beta_w0():
    reference_type = 'theta_beta_w0'
    ref_file = os.path.join(REFERENCE_PATH, reference_type, REFERENCE_FILE)
    assert os.path.exists(ref_file), "No reference file found"
    reference = read_yaml(ref_file)

    test_cases = ['theta_180_beta_0_w0_1',
                  'theta_145_beta_45_w0_2',
                  'theta_90_beta_90_w0_4']

    for case in test_cases:
        vars = case.split('_')
        theta, beta, w0_factor = [int(var) for var in vars[1::2]]

        tau = 25e-15
        W = 25
        lam = 0.8e-6
        w0 = w0_factor*lam
        
        for mode in 'explicit solver'.split():
            ref_result = reference[mode][case]['vacem']
            quvac_mode = 'analytic' if mode == 'explicit' else 'maxwell'

            # Define fields
            field_1 = {
                "field_type": f"paraxial_gaussian_{quvac_mode}",
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
                "field_type": f"paraxial_gaussian_{quvac_mode}",
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
                'performance': {}
            }

            path = f'data/test/test_alex_reference/{mode}/{case}'
            Path(path).mkdir(parents=True, exist_ok=True)

            ini_file = os.path.join(path, 'ini.yml')
            write_yaml(ini_file, ini_data)

            # Launch simulation
            status = os.system(f"{SCRIPT_PATH} --input {ini_file}")
            assert status == 0, "Script execution did not finish successfully"

            result_file = os.path.join(path, 'spectra.npz')
            quvac_data = np.load(result_file)
            quvac_result = quvac_data['N_total']

            print(f"Total signal (quvac): {quvac_result:.4f}")
            print(f"Total signal (vacem): {ref_result:.4f}")

            err_msg = 'quvac and vacem results differ by more than 0.5%'
            assert np.isclose(quvac_result, ref_result, rtol=5e-3), err_msg





