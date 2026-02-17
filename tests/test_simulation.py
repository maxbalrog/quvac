"""
Collection of tests for simulations.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c, hbar, pi

from quvac.postprocess import (
    get_simulation_fields,
    get_spectra_from_Stokes,
    integrate_spherical,
    signal_in_detector,
)
from quvac.simulation import quvac_simulation
from quvac.simulation_parallel import quvac_simulation_parallel
from quvac.utils import read_yaml, write_yaml
from tests.config_for_tests import DEFAULT_CONFIG_PATH

##########################################################################################
# SIMULATION
##########################################################################################


@pytest.fixture(scope="session")
def get_tmp_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("simulation")
    return tmp_path


def save_ini(path, ini_data):
    path = os.path.join(str(path), "test_simulation")
    Path(path).mkdir(parents=True, exist_ok=True)

    ini_file = os.path.join(path, "ini.yml")
    write_yaml(ini_file, ini_data)
    return ini_file


def test_simulation(tmp_path):
    ini_data = read_yaml(DEFAULT_CONFIG_PATH)
    ini_file = save_ini(tmp_path, ini_data)
    quvac_simulation(ini_file)

    ini_data["mode"] = "postprocess"
    ini_file = save_ini(tmp_path, ini_data)
    quvac_simulation(ini_file)


def test_simulation_test(tmp_path):
    ini_data = read_yaml(DEFAULT_CONFIG_PATH)
    ini_data["performance"]["test_run"] = True
    ini_file = save_ini(tmp_path, ini_data)
    quvac_simulation(ini_file)


def test_channels(tmp_path):
    ini_data = read_yaml(DEFAULT_CONFIG_PATH)
    ini_data["integrator"]["type"] = "vacuum_emission_channels"
    ini_file = save_ini(tmp_path, ini_data)
    quvac_simulation(ini_file)


def test_precision(tmp_path):
    ini_data = read_yaml(DEFAULT_CONFIG_PATH)
    ini_data["performance"]["precision"] = "float32"
    ini_data["postprocess"]["perp_polarization_type"] = None
    ini_file = save_ini(tmp_path, ini_data)
    quvac_simulation(ini_file)


##########################################################################################
# POSTPROCESS
##########################################################################################


def compare_total_number_of_bg_photons(ini_file):
    folder = os.path.dirname(ini_file)
    data = np.load(os.path.join(folder, "spectra_total.npz"))
    k,theta,phi,background = [
        data[key] for key in "k theta phi background".split()
    ]
    bg_total_num = integrate_spherical(
        background, 
        (theta,phi),
        axs_integrate=["theta","phi"],
        axs_names=["theta", "phi"]
    )

    ini_data = read_yaml(ini_file)
    bg_total_estimate = 0
    for field in ini_data["fields"].values():
        W, lam = field["W"], field["lam"]
        bg_total_estimate += W*lam / (2*pi*c*hbar)

    err_msg = ("Total number of bg photons from numerical calculation is not close to"
              f"physics estimation! total (num): {bg_total_num:.2e}," 
              f"total (est): {bg_total_estimate:.2e}")
    assert np.isclose(bg_total_num, bg_total_estimate, rtol=0.5), err_msg


def test_postprocess(tmp_path):
    ini_data = read_yaml(DEFAULT_CONFIG_PATH)
    # test spherical and discernible signal
    ini_data["postprocess"].update({
        "calculate_spherical": True,
        "calculate_discernible": True,
        "discernibility": "angular",
        "perp_polarization_type": "local axis",
    })
    ini_file = save_ini(tmp_path, ini_data)
    quvac_simulation(ini_file)

    # test that total number of bg photons agrees with a simple physical estimate
    compare_total_number_of_bg_photons(ini_file)

    # test stokes, xyz_background
    ini_data["mode"] = "postprocess"
    ini_data["postprocess"].update({
        "calculate_xyz_background": True,
        "bgr_idx": 1,
        "stokes": True,
    })
    ini_file = save_ini(tmp_path, ini_data)
    quvac_simulation(ini_file)

    # test loading stokes parameters
    folder = os.path.dirname(ini_file)
    data = np.load(os.path.join(folder, "spectra_total.npz"))
    P0 = data["N_xyz"]
    data_p = np.load(os.path.join(folder, "spectra_polarization.npz"))
    P1, P2, P3 = [data_p[key] for key in "P1 P2 P3".split()]
    for basis in "linear linear-45 circular".split():
        Nf, Np = get_spectra_from_Stokes(P0, P1, P2, P3, basis=basis)
        assert np.allclose(Nf + Np, P0), "Total signal should be equal to the summ of"
        " perp polarization modes."

    # test loading fields from ini_file
    fields = get_simulation_fields(ini_file)
    assert len(fields) == 2, "There should be 2 fields in the default simulation."

    # test signal in detector
    N_sph, k, theta, phi = [data[key] for key in "N_sph k theta phi".split()]
    N_angular = integrate_spherical(N_sph, (k,theta,phi), axs_integrate=['k'])
    detector = {
        "theta0": 0,
        "phi0": 0,
        "dtheta": 15,
        "dphi": 15,
    }
    N_detector = signal_in_detector(N_angular, theta, phi, detector, align_to_max=True)
    assert N_detector > 0, "Signal should be positive"


def test_mix_bg_and_signal(tmp_path):
    ini_data = read_yaml(DEFAULT_CONFIG_PATH)
    ini_data["postprocess"].update({
        "modes": ["mix_signal_bg"],
        "add_signal_bg": True,
    })
    ini_file = save_ini(tmp_path, ini_data)
    quvac_simulation(ini_file)


##########################################################################################
# SIMULATION PARALLEL
##########################################################################################


def test_parallel_simulation(tmp_path):
    # run usual simulation
    ini_data = read_yaml(DEFAULT_CONFIG_PATH)
    ini_file = save_ini(tmp_path, ini_data)
    quvac_simulation(ini_file)
    
    folder = os.path.dirname(ini_file)
    data = np.load(os.path.join(folder, "spectra_total.npz"))
    N_total = data["N_total"]

    ini_data["performance"]["nthreads"] = 2
    ini_file = save_ini(tmp_path, ini_data)
    quvac_simulation_parallel(ini_file)
    data = np.load(os.path.join(folder, "spectra_total.npz"))
    N_total_parallel = data["N_total"]

    assert np.isclose(N_total, N_total_parallel), "Sequential and parallel results "
    "should be the same."


