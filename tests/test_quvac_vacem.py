"""
Here we provide a comparison between quvac and vacem results
"""

import os
from pathlib import Path

from config_for_tests import DEFAULT_CONFIG_PATH, SIMULATION_SCRIPT
import numpy as np
import pytest

from quvac.utils import read_yaml, write_yaml

REFERENCE_PATH = "tests/references"
REFERENCE_FILE = "results.yml"


@pytest.mark.slow
def test_theta_beta_w0():
    reference_type = "theta_beta_w0"
    ref_file = os.path.join(REFERENCE_PATH, reference_type, REFERENCE_FILE)
    assert os.path.exists(ref_file), "No reference file found"
    reference = read_yaml(ref_file)

    test_cases = [
        "theta_180_beta_0_w0_1",
        "theta_145_beta_45_w0_2",
        "theta_90_beta_90_w0_4",
    ]

    ini_data = read_yaml(DEFAULT_CONFIG_PATH)

    for case in test_cases:
        vars = case.split("_")
        theta, beta, w0_factor = [int(var) for var in vars[1::2]]

        lam = 0.8e-6
        w0 = w0_factor * lam

        field_1_params = {
            "w0": w0,
        }
        field_2_params = {"w0": w0, "theta": theta, "beta": beta}
        ini_data["fields"]["field_1"].update(field_1_params)
        ini_data["fields"]["field_2"].update(field_2_params)

        for mode in "explicit solver".split():
            ref_result = reference[mode][case]["vacem"]
            quvac_mode = "analytic" if mode == "explicit" else "maxwell"

            field_type = f"paraxial_gaussian_{quvac_mode}"
            ini_data["fields"]["field_1"]["field_type"] = field_type
            ini_data["fields"]["field_2"]["field_type"] = field_type

            path = f"data/test/test_alex_reference/{mode}/{case}"
            Path(path).mkdir(parents=True, exist_ok=True)

            ini_file = os.path.join(path, "ini.yml")
            write_yaml(ini_file, ini_data)

            # Launch simulation
            status = os.system(f"{SIMULATION_SCRIPT} --input {ini_file}")
            assert status == 0, "Script execution did not finish successfully"

            result_file = os.path.join(path, "spectra_total.npz")
            quvac_data = np.load(result_file)
            quvac_result = quvac_data["N_total"]

            print(f"Total signal (quvac): {quvac_result:.4f}")
            print(f"Total signal (vacem): {ref_result:.4f}")

            err_msg = "quvac and vacem results differ by more than 10%"
            assert np.isclose(quvac_result, ref_result, rtol=1e-1), err_msg
