"""
Collection of tests for numerical grid creation.
"""
from copy import deepcopy

import numpy as np
import pytest

from quvac.grid import (
    GridXYZ,
    create_dynamic_grid,
    get_box_size,
    get_bw,
    get_ek,
    get_kmax_grid,
    get_pol_basis,
    get_t_size,
    get_xyz_size,
    setup_grids,
)

TESTED_FIELD_TYPES = ["gaussian", "dipole"]

@pytest.mark.parametrize(
    ("theta", "phi", "expected_ek", "expected_e1", "expected_e2"),
    [
        ( 0, 0,np.array([0,0,1]),np.array([1,0,0]),np.array([0,1,0])),
        (90, 0,np.array([1,0,0]),np.array([0,0,-1]),np.array([0,1,0])),
        (90,90,np.array([0,1,0]),np.array([0,0,-1]),np.array([-1,0,0]))
    ],
)
def test_ek_and_pol_basis(theta, phi, expected_ek, expected_e1, expected_e2):
    theta, phi = np.radians(theta), np.radians(phi)
    ek = get_ek(theta, phi)
    e1, e2 = get_pol_basis(theta, phi)
    assert np.allclose(ek, expected_ek)
    assert np.allclose(e1, expected_e1)
    assert np.allclose(e2, expected_e2)


def get_default_field_params(field_type="gaussian"):
    field_params = {
        "field_type": f"{field_type}_maxwell",
        "lam": 800e-9,
        "tau": 25e-15,
        "theta": 90,
        "phi": 0,
    }
    if field_type == "gaussian":
        field_params["w0"] = 4*800e-9
    return field_params


@pytest.mark.parametrize("field_type", TESTED_FIELD_TYPES)
def test_bandwidth_ok_field_params(field_type):
    field_params = get_default_field_params(field_type)
    bw_perp, bw_long = get_bw(field_params)
    assert bw_perp > 0, "Perpendicular bandwidth should be positive."
    assert bw_long > 0, "Longitudinal bandwidth should be positive."


def test_bandwidth_missing_gauss_params():
    field_params = get_default_field_params("gaussian")
    field_params.pop("w0")
    with pytest.raises(AssertionError):
        get_bw(field_params)


@pytest.mark.parametrize("field_type", TESTED_FIELD_TYPES)
def test_get_kmax_grid(field_type):
    field_params = get_default_field_params(field_type)
    kmax = get_kmax_grid(field_params)
    assert np.all(kmax > 0), "kmax should be positive."
    assert kmax.shape == (3,), "kmax should be a 3-element array."


def test_get_kmax_grid_invalid_field_type():
    field_params = get_default_field_params("invalid_field")
    with pytest.raises(NotImplementedError):
        get_kmax_grid(field_params)


@pytest.mark.parametrize("missing_param", ["lam", "tau", "theta", "phi"])
def test_get_kmax_grid_missing_field_params(missing_param):
    field_params = get_default_field_params("gaussian")
    field_params.pop(missing_param)
    with pytest.raises(AssertionError):
        get_kmax_grid(field_params)


def test_get_t_size():
    Nt = get_t_size(0, 25e-15, lam=800e-9)
    assert Nt > 0, "Nt should be positive."
    assert isinstance(Nt, int), "Nt should be an integer."
    Nt_2 = get_t_size(0, 25e-15, lam=800e-9, grid_res=2)
    assert Nt_2 == 2*Nt-1, "Nt should be double when grid_res is 2."


@pytest.mark.parametrize("field_type", TESTED_FIELD_TYPES)
def test_get_xyz_size(field_type):
    field_params = get_default_field_params(field_type)
    box_size = [10e-6, 10e-6, 10e-6]
    Nxyz = get_xyz_size([field_params], box_size)
    assert isinstance(Nxyz, list), "Nxyz should be a list."
    assert len(Nxyz) == 3, "Nxyz should have 3 elements."

    # test with double grid resolution
    Nxyz_2 = get_xyz_size([field_params], box_size, grid_res=2)
    Nxyz_2_expected_1 = [2*n for n in Nxyz]
    Nxyz_2_expected_2 = deepcopy(Nxyz_2_expected_1)
    Nxyz_2_expected_2[0] -= 1
    condition = (Nxyz_2 == Nxyz_2_expected_1) or (Nxyz_2 == Nxyz_2_expected_2)
    assert condition, "Nxyz should be double when grid_res is 2."

    # test with equal resolution
    Nxyz = get_xyz_size([field_params], box_size, equal_resolution=True)
    assert (Nxyz[0] == Nxyz[1]) and (Nxyz[1] == Nxyz[2]), "grid should be equal in all "
    "dimensions."


def get_default_grid_params():
    return {
        "mode": "dynamic",
        "geometry": "x",
        "transverse_factor": 20,
        "longitudinal_factor": 8,
        "time_factor": 2,
    }


@pytest.mark.parametrize("field_type", TESTED_FIELD_TYPES)
def test_get_box_size(field_type):
    field_params = get_default_field_params(field_type)
    grid_params = get_default_grid_params()
    trans, long = get_box_size([field_params], grid_params)
    assert isinstance(trans, float), "Transverse size should be a float."
    assert isinstance(long, float), "Longitudinal size should be a float."


@pytest.mark.parametrize("field_type", TESTED_FIELD_TYPES)
def test_create_dynamic_grid(field_type):
    field_params = get_default_field_params(field_type)
    grid_params = get_default_grid_params()
    grid_params_upd = create_dynamic_grid([field_params], grid_params)
    new_grid_keys = "box_xyz Nxyz box_t Nt".split()
    assert isinstance(grid_params_upd, dict), "grid_params should be a dictionary."
    assert all(key in grid_params_upd for key in new_grid_keys), "grid_params should "
    "contain box_xyz, Nxyz, box_t and Nt keys."

    # test with iterable  grid resolution
    grid_params["spatial_resolution"] = [1, 2, 3]
    grid_params_upd = create_dynamic_grid([field_params], grid_params)
    assert all(key in grid_params_upd for key in new_grid_keys), "grid_params should "
    "contain box_xyz, Nxyz, box_t and Nt keys."


@pytest.fixture
def sample_grid():
    return [
        np.linspace(-5e-6, 5e-6, 200),
        np.linspace(-5e-6, 5e-6, 100),
        np.linspace(-5e-6, 5e-6, 100),
    ]


def test_GridXYZ(sample_grid):
    grid = GridXYZ(sample_grid)
    assert grid.grid_shape == (200, 100, 100), "Grid shape should match input."
    assert grid.grid == sample_grid, "Grid should match input."
    assert grid.kgrid is None, "kgrid should be None by default."

    grid.get_k_grid()
    kgrid_shape = tuple(len(k) for k in grid.kgrid)
    k_keys = "kx ky kz kgrid kgrid_shifted e1x e1y e1z e2x e2y e2z".split()
    assert kgrid_shape == (200, 100, 100), "kgrid shape should match input grid shape."
    assert all(hasattr(grid, key) for key in k_keys), "Grid should contain kgrid keys."


def get_two_default_gaussians():
    field_1 = {
        "field_type": "paraxial_gaussian_maxwell",
        "lam": 800e-9,
        "tau": 25e-15,
        "w0": 4*800e-9,
        "theta": 90,
        "phi": 0,
    }
    field_2 = deepcopy(field_1)
    field_2["phi"] = 180
    return [field_1, field_2]


def test_setup_grids():
    fields = get_two_default_gaussians()
    grid_params = get_default_grid_params()
    grid, grid_t = setup_grids(fields, grid_params)

    grid_params = create_dynamic_grid(fields, grid_params)
    box_t = grid_params["time_factor"] * fields[0]["tau"]
    grid_params["mode"] = "direct"
    grid_params["box_t"] = [-box_t, box_t]
    grid_2, grid_t_2 = setup_grids(fields, grid_params)

    assert np.allclose(grid_t, grid_t_2), "Time grid should match."
    condition = all(np.allclose(g1, g2) for g1, g2 in zip(grid.grid, grid_2.grid))  # noqa: B905
    assert condition, "Spatial grids should match."


def test_ignore_idx():
    fields = get_two_default_gaussians()
    grid_params = get_default_grid_params()
    grid_params["ignore_idx"] = [1]
    grid, grid_t = setup_grids(fields, grid_params)


