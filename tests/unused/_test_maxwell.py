"""
Here we provide a test for maxwell representation of fields:
for some field configurations we compare analytic and maxwell fields
"""

from config_for_tests import DEFAULT_CONFIG_PATH
import numpy as np

from quvac.field.gaussian import GaussianAnalytic
from quvac.field.maxwell import MaxwellMultiple
from quvac.grid import setup_grids
from quvac.utils import read_yaml


def get_intensity(field, t):
    E, B = field.calculate_field(t=t)
    E, B = [np.real(Ex) for Ex in E], [np.real(Bx) for Bx in B]
    intensity = (E[0]**2 + E[1]**2 + E[2]**2 + B[0]**2 + B[1]**2 + B[2]**2)/2
    return intensity


def test_maxwell_gauss():
    ini_data = read_yaml(DEFAULT_CONFIG_PATH)
    for idx in [1,2]:
        key = f"field_{idx}"
        ini_data["fields"][key]["w0"] = 4 * ini_data["fields"][key]["lam"]

    field_params = ini_data["fields"]["field_1"]
    grid_params = ini_data["grid"]

    grid_xyz, grid_t = setup_grids([field_params], grid_params)
    grid_xyz.get_k_grid()
    
    gauss = GaussianAnalytic(field_params, grid_xyz)
    gauss_mw = MaxwellMultiple([field_params], grid_xyz)

    t = 0.
    intensity = get_intensity(gauss, t)
    intensity_mw = get_intensity(gauss_mw, t)

    intensity = np.clip(intensity, a_min=intensity.max()*1e-6, a_max=None)
    intensity_mw = np.clip(intensity_mw, a_min=intensity.max()*1e-6, a_max=None)

    err_msg = "Maxwell field does not match analytic field (up to 20% relative error)"
    assert np.allclose(intensity, intensity_mw, rtol=2e-1), err_msg