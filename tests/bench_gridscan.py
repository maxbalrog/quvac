"""
Test for quvac gridscan with script.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from quvac.utils import read_yaml, write_yaml
from tests.config_for_tests import BENCHMARK_CONFIG_PATH, GRIDSCAN_SCRIPT


@pytest.mark.benchmark
def test_gridscan():
    # Define field parameters
    ini_data = read_yaml(BENCHMARK_CONFIG_PATH)

    beta_arr = [0, 45, 90]
    variables_data = {
        "create_grids": True,
        "fields": {"field_2": {"beta": [0, 90, 3]}},
        "cluster": {
            "cluster": "local",
            "max_parallel_jobs": 10,
        },
    }

    path = "data/test/test_gridscan"
    Path(path).mkdir(parents=True, exist_ok=True)

    ini_file = os.path.join(path, "ini.yaml")
    ini_data["variables"] = variables_data
    write_yaml(ini_file, ini_data)

    # Launch simulation
    status = os.system(
        f"{GRIDSCAN_SCRIPT} --input {ini_file}"
    )
    assert status == 0, "Script execution did not finish successfully"

    folders = [f"#field_2:beta_{beta}" for beta in beta_arr]
    data = []
    for folder in folders:
        data_loc = np.load(os.path.join(path, folder, "spectra_total.npz"))
        data.append(data_loc["N_total"])

    err_msg = "Calculated results do not agree with analytics"
    assert np.isclose(data[1] / data[0], 130 / 64, rtol=1e-1), err_msg
