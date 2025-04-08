"""
Test for oftimization of quvac simulations.
"""

import os
from pathlib import Path

import numpy as np
import pytest

try:
    from ax.service.ax_client import AxClient
except ModuleNotFoundError as exc:
    print("`ax` package should be installed to use optimization")
    raise ModuleNotFoundError("`ax` package not found") from exc

from quvac.cluster.optimization import gather_trials_data
from quvac.utils import read_yaml, write_yaml
from tests.config_for_tests import BENCHMARK_CONFIG_PATH, OPTIMIZATION_SCRIPT


@pytest.mark.slow
@pytest.mark.benchmark
def test_optimization():
    ini_data = read_yaml(BENCHMARK_CONFIG_PATH)

    path = "data/test/test_optimization"
    Path(path).mkdir(parents=True, exist_ok=True)

    # Modify some parameters
    ini_data["postprocess"]["calculate_spherical"] = True
    ini_data["postprocess"]["calculate_discernible"] = True
    ini_data["scales"] = {}
    ini_data["save_path"] = path

    optimization_data = {
        "name": "2_pulses_beta",
        "parameters": {"field_2": {"beta": [0, 90]}},
        "cluster": {
            "cluster": "local",
        },
        "n_trials": 10,
        "objectives": [["N_total", False]],
    }

    ini_file = os.path.join(path, "ini.yml")
    write_yaml(ini_file, ini_data)

    optimization_file = os.path.join(path, "optimization.yml")
    write_yaml(optimization_file, optimization_data)

    # Launch simulation
    status = os.system(
        f"{OPTIMIZATION_SCRIPT} --input {ini_file} --optimization {optimization_file}"
    )
    assert status == 0, "Script execution did not finish successfully"

    client_json = os.path.join(path, "experiment.json")
    ax_client = AxClient.load_from_json_file(client_json)

    trials_params = gather_trials_data(ax_client, metric_names=["N_total"])
    betas = [val["field_2:beta"] for val in trials_params.values()]
    N_total = [val["N_total"] for val in trials_params.values()]
    beta_max = betas[np.argmax(N_total)]

    err_msg = "Optimization-found value of beta is not close to the real optimum"
    assert np.isclose(beta_max, 90.0, rtol=1e-1), err_msg
