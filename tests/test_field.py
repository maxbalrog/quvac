"""
Collection of tests for fields.
"""
import numpy as np
import pytest

from quvac.field.dipole import DipoleAnalytic, create_multibeam
from quvac.field.external_field import ExternalField, ProbePumpField
from quvac.field.gaussian import GaussianAnalytic
from quvac.field.maxwell import MaxwellMultiple
from quvac.field.model import EBInhomogeneity
from quvac.field.utils import convert_tau, get_intensity, get_max_edge_amplitude
from quvac.grid import setup_grids

##########################################################################################
# GAUSSIAN FIELD
##########################################################################################

def get_default_gaussian_params(field_mode="maxwell"):
    return {
        "field_type": f"paraxial_gaussian_{field_mode}",
        "focus_x": [0.,0.,0.],
        "focus_t": 0.,
        "theta": 0,
        "phi": 0,
        "beta": 0,
        "lam": 800e-9,
        "w0": 4*800e-9,
        "tau": 25e-15,
        "W": 25,
        "phase0": 0,
        "order": 0,
    }


def get_default_grid_params():
    return {
        "mode": "dynamic",
        "geometry": "z",
        "transverse_factor": 10,
        "longitudinal_factor": 6,
        "time_factor": 2,
    }


def get_gaussian(order=0):
    gauss_params = get_default_gaussian_params("maxwell")
    gauss_params["order"] = order
    grid_params = get_default_grid_params()
    grid, _ = setup_grids([gauss_params], grid_params)
    field = GaussianAnalytic(gauss_params, grid)
    return field


def check_fields(E, B, check_dtype=True, mode="real"):
    for idx in range(3):
        condition = E[idx].shape == B[idx].shape
        assert condition, "Electric and magnetic field array shapes should be equal."
    if check_dtype:
        for idx in range(3):
            dtype = np.float64 if mode == "real" else np.complex128
            condition = E[idx].dtype == dtype
            assert condition, "Field should have appropriate data type."


@pytest.mark.parametrize("order, mode", [(0,"real"), (5,"real"), (0,"complex")])
def test_gaussian(order, mode):
    gaussian = get_gaussian(order)
    E,B = gaussian.calculate_field(t=0., mode=mode)
    check_fields(E, B, mode=mode)


def get_default_grid(params):
    grid_params = get_default_grid_params()
    if params[0]["field_type"].startswith("dipole"):
        grid_params["transverse_factor"] = 6
    grid, _ = setup_grids(params, grid_params)
    grid.get_k_grid()
    return grid


@pytest.fixture
def gaussian_maxwell():
    params = [get_default_gaussian_params("maxwell")]
    grid = get_default_grid(params)
    field = ExternalField(params, grid)
    return field


def test_gaussian_maxwell(gaussian_maxwell):
    E,B = gaussian_maxwell.calculate_field(t=0.)
    check_fields(E, B, check_dtype=False)


def test_incorrect_field_type_for_maxwell():
    params = [get_default_gaussian_params("incorrect_type")]
    grid = get_default_grid(params)
    with pytest.raises(NotImplementedError):
        ExternalField(params, grid)
    
    with pytest.raises(NotImplementedError):
        MaxwellMultiple(params, grid)


def test_zero_energy_field():
    params = get_default_gaussian_params("maxwell")
    params["W"] = 0
    grid = get_default_grid([params])
    field = MaxwellMultiple([params], grid)
    condition = np.allclose(field.a1, 0.) and np.allclose(field.a2, 0.)
    assert condition, "Spectral coefficients should be zero for W=0"


def get_two_default_gaussians():
    g1 = get_default_gaussian_params("maxwell")
    g2 = get_default_gaussian_params("analytic")
    g2["theta"] = 180
    return [g1, g2]


def test_pump_probe():
    fields = get_two_default_gaussians()
    grid = get_default_grid(fields)
    field = ProbePumpField(fields, grid)
    field.calculate_field(t=0.)


##########################################################################################
# DIPOLE FIELD
##########################################################################################


def get_default_dipole(field_mode="maxwell", envelope="gauss", dipole_type="electric"):
    return {
        "field_type": f"dipole_{field_mode}",
        "envelope": envelope,
        "dipole_type": dipole_type,
        "focus_x": [0.,0.,0.],
        "focus_t": 0.,
        "theta": 0,
        "phi": 0,
        "lam": 800e-9,
        "tau": 25e-15,
        "W": 25,
    }


@pytest.mark.parametrize("envelope, dipole_type",
                         [("gauss", "electric"),("plane", "magnetic")])
def test_dipole(envelope, dipole_type):
    params = get_default_dipole(envelope=envelope, dipole_type=dipole_type)
    grid = get_default_grid([params])
    field = DipoleAnalytic(params, grid)
    E,B = field.calculate_field(t=0.)
    check_fields(E, B, mode="real")


def test_dipole_incorrect_envelope():
    params = get_default_dipole(envelope="incorrect_type")
    grid = get_default_grid([params])
    with pytest.raises(NotImplementedError):
        DipoleAnalytic(params, grid)


@pytest.mark.parametrize("n_beams, multibeam_mode",
                         [(4,"belt"), (6,"belt"), 
                          (12,"sphere")])
def test_multibeam(n_beams, multibeam_mode):
    # create seed beam
    field_seed = get_default_gaussian_params()
    field_seed.update({
        "theta": 90,
        "phi": 0,
    })

    if multibeam_mode == "belt":
        phi0 = 360/(n_beams*2)
    elif multibeam_mode == "sphere":
        phi0 = 360/(n_beams//3*2)

    multibeam_params = create_multibeam(field_seed, n_beams=n_beams,
                                        mode=multibeam_mode, phi0=phi0)
    assert len(multibeam_params) == n_beams, f"Multibeam should consist of {n_beams} "
    f"beams but it has {len(multibeam_params)}"


##########################################################################################
# MODEL FIELD
##########################################################################################


def get_inhom_params(field_inhom, envelope):
    return {
        "field_type": "eb_inhomogeneity",
        "field_inhom": field_inhom,
        "envelope_type": envelope,
        "theta": 90,
        "phi": 90,
        "W": 1,
        "w0": 2*800e-9,
        "lam": 800e-9,
    }


@pytest.mark.parametrize("field_inhom, envelope",
                         [("magnetic","gauss"), ("electric","gauss-modulated")])
def test_model_fields(field_inhom, envelope):
    inhom_params = get_inhom_params(field_inhom, envelope)
    gauss_params = get_default_gaussian_params()
    grid = get_default_grid([gauss_params])

    field = EBInhomogeneity(inhom_params, grid)
    E,B = field.calculate_field(t=0.)
    check_fields(E, B, mode="real")


def test_incorrect_model_fields():
    gauss_params = get_default_gaussian_params()
    grid = get_default_grid([gauss_params])

    inhom_params_1 = get_inhom_params("incorrect_type", "gauss")
    inhom_params_2 = get_inhom_params("electric", "incorrect_type")
    for inhom_params in [inhom_params_1, inhom_params_2]:
        with pytest.raises(NotImplementedError):
            EBInhomogeneity(inhom_params, grid)


##########################################################################################
# FIELD UTILS
##########################################################################################


def test_convert_tau():
    # it is our reference
    tau = 25
    taus_ref = convert_tau(tau, mode="1/e^2")
    # check with other conventions
    modes = ["FWHM", "FWHM-Intensity", "std"]
    for mode in modes:
        taus = convert_tau(taus_ref[mode], mode=mode)
        condition = all(np.isclose(t1,t2) for t1,t2 
                        in zip(taus_ref.values(), taus.values()))  # noqa: B905
        assert condition, "Results for different conventions should be the same."

    # check with incorrect convention name
    with pytest.raises(NotImplementedError):
        convert_tau(tau, mode="incorrect_type")


def test_intensity(gaussian_maxwell):
    intensity = get_intensity(gaussian_maxwell, t=0.)
    assert np.all(intensity > 0), "Intensity should be >0."

    E,_ = gaussian_maxwell.calculate_field(t=0.)
    amplitude_edge = get_max_edge_amplitude(E)
    assert np.all(amplitude_edge > 0), "Field amplitude on edges should be >0."

