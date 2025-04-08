"""
Here we provide a test for quantum vacuum signal calculator
with script in several scenarios:
1) test_simulation: default collision scenario
2) test_mixed_fields: one field is analytic, another is maxwell
3) test_spherical: interpolation of signal spectrum on spherical grid
4) test_discernible: calculate discernible signal and compare that it is 
                     close to paper result
5) test_channels: test channel separation in the integrator
"""

import os
from pathlib import Path

import numpy as np
import pytest

from quvac.analytic_scalings import get_two_paraxial_scaling
from quvac.utils import read_yaml, write_yaml
from tests.config_for_tests import BENCHMARK_CONFIG_PATH, SIMULATION_SCRIPT


@pytest.mark.benchmark
def run_test_simulation(path, ini_data):
    Path(path).mkdir(parents=True, exist_ok=True)

    ini_file = os.path.join(path, "ini.yml")
    write_yaml(ini_file, ini_data)

    # Launch simulation
    status = os.system(f"{SIMULATION_SCRIPT} --input {ini_file}")
    assert status == 0, "Script execution did not finish successfully"


@pytest.mark.benchmark
def test_simulation():
    # Load default simulation parameters
    ini_data = read_yaml(BENCHMARK_CONFIG_PATH)

    path = "data/test/test_simulation"
    run_test_simulation(path, ini_data)


@pytest.mark.benchmark
def test_compare_with_analytics():
    # Load default simulation parameters
    ini_data = read_yaml(BENCHMARK_CONFIG_PATH)
    field_1_params = {
        "field_type": "paraxial_gaussian_analytic",
        "w0": 2 * 0.8e-6,
    }
    field_2_params = {
        "field_type": "paraxial_gaussian_analytic",
        "w0": 2 * 0.8e-6,
        "theta": 90,
        "beta": 90,
    }
    ini_data["fields"]["field_1"].update(field_1_params)
    ini_data["fields"]["field_2"].update(field_2_params)
    ini_data["grid"]["collision_geometry"] = "xz"

    path = "data/test/test_compare_with_analytics"
    run_test_simulation(path, ini_data)

    data = np.load(os.path.join(path, "spectra_total.npz"))
    N_signal_num = data["N_total"]

    fields = list(ini_data["fields"].values())
    N_signal_th, N_perp_th = get_two_paraxial_scaling(fields)

    err_msg = "Analytical and numerical total signal differ by more than 1%"
    assert np.isclose(N_signal_th, N_signal_num, rtol=1e-2), err_msg


@pytest.mark.benchmark
def test_mixed_fields():
    # Load default simulation parameters
    ini_data = read_yaml(BENCHMARK_CONFIG_PATH)
    ini_data["fields"]["field_1"]["field_type"] = "paraxial_gaussian_analytic"

    path = "data/test/test_mixed_fields"
    run_test_simulation(path, ini_data)


@pytest.mark.benchmark
def test_spherical_grid():
    # Load default simulation parameters
    ini_data = read_yaml(BENCHMARK_CONFIG_PATH)
    ini_data["postprocess"]["calculate_spherical"] = True
    ini_data["postprocess"]["calculate_discernible"] = False

    path = "data/test/test_spherical_grid"
    run_test_simulation(path, ini_data)

    data = np.load(os.path.join(path, "spectra_total.npz"))
    err_msg = "Total signal on cartesian and spherical grid differ by more than 1%"
    assert np.isclose(data["N_total"], data["N_sph_total"], rtol=1e-2), err_msg


@pytest.mark.slow
@pytest.mark.benchmark
def test_discernible():
    # Load default simulation parameters
    ini_data = read_yaml(BENCHMARK_CONFIG_PATH)
    # Change field parameters
    field_2_params = {"w0": 4 * 0.8e-6, "theta": 160, "beta": 90}
    ini_data["fields"]["field_2"].update(field_2_params)
    ini_data["postprocess"]["calculate_spherical"] = True
    ini_data["postprocess"]["calculate_discernible"] = True

    path = "data/test/test_discernible"
    run_test_simulation(path, ini_data)

    data = np.load(os.path.join(path, "spectra_total.npz"))
    N_disc = data["N_disc"]
    N_disc_expected = 2.5
    err_msg = "Calculated discernible signal differs from expected by more than 20%"
    assert np.isclose(N_disc, N_disc_expected, rtol=2e-1), err_msg


@pytest.mark.benchmark
def test_channels():
    # Load default simulation parameters
    ini_data = read_yaml(BENCHMARK_CONFIG_PATH)

    results = []
    for idx, channels in zip([(0, 0), (0, 1), (1, 0)], [False, True, True]):  # noqa: B905
        integrator_type = "vacuum_emission_channels" if channels else "vacuum_emission"

        ini_data["integrator"] = {
            "type": integrator_type,
            "probe_pump_idx": {
                "probe": [idx[0]],
                "pump": [idx[1]],
            },
        }

        path = f"data/test/test_channels_{idx[0]}_{idx[1]}"
        run_test_simulation(path, ini_data)

        data_file = os.path.join(path, "spectra_total.npz")
        data = np.load(data_file)

        N_signal_num = data["N_total"]
        results.append(N_signal_num)

    # Test that signal_channel_1 + signal_channel_2 = total_signal
    # and signal_channel_1 == signal_channel_2
    err_msg = "Signal is different in two channels"
    assert np.isclose(results[1], results[2], rtol=1e-5), err_msg
    err_msg = "Signal in two channels does not add up to the total signal"
    assert np.isclose(results[0], results[1] + results[2], rtol=1e-2), err_msg
