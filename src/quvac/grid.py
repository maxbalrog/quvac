"""
Functions for creating spatial and temporal grids.

1. ``GridXYZ`` class that calculates the spatial grid and its 
Fourier counterpart.

2. Helper function ``setup_grids`` that automatically creates 
grids either dynamically or from given grid sizes.

.. warning::
    The dynamic grid creation is experimental and should be used with caution.
    Especially, when testing on new physical problems, it is recommended to
    visualize participating Maxwell fields and check the grid sizes.

.. note::
    Parts of dynamical grid creation for gaussian beams were initially
    implemented by Alexander Blinne (``get_kmax_from_bw``, ``get_xyz_size``,
    ``get_t_size``).
"""

from collections.abc import Iterable
from copy import deepcopy

import numexpr as ne
import numpy as np
import pyfftw
from scipy.constants import c, pi

from quvac import config


class GridXYZ:
    """
    Calculates spatial grid and its Fourier counterpart.

    Parameters
    ----------
    grid : tuple of np.array
        A tuple containing three numpy arrays (x, y, z) representing the spatial grid 
        for field discretization.
    
    Attributes
    ----------
    grid : tuple of np.array
        The spatial grid for field discretization.
    grid_shape : tuple of int
        The shape of the grid.
    xyz : tuple of np.array
        The meshgrid of the spatial grid.
    dxyz : tuple of float
        The grid spacing in each dimension.
    dV : float
        The volume element of the grid.
    kgrid : tuple of np.array
        The k-space grid.
    kgrid_shifted : tuple of np.array
        The shifted k-space grid.
    dkxkykz : list of float
        The spacing in k-space for each dimension.
    dVk : float
        The volume element in k-space.
    kmeshgrid : tuple of np.array
        The meshgrid of the k-space grid.
    kabs : np.array
        The absolute value of the k-vectors.
    e1x, e1y, e1z : np.array
        The x, y, and z components of the first polarization vector.
    e2x, e2y, e2z : np.array
        The x, y, and z components of the second polarization vector.
    """

    def __init__(self, grid):
        # Define spatial grid
        self.grid = grid
        self.grid_shape = tuple(ax.size for ax in grid)
        self.xyz = self.x, self.y, self.z = np.meshgrid(
            *grid, indexing="ij", sparse=True
        )
        self.dxyz = tuple(ax[1] - ax[0] for ax in grid)
        self.dV = np.prod(self.dxyz)
        self.kgrid = None

    def get_k_grid(self):
        """
        Calculate the k-space grid and polarization vectors.

        Returns
        -------
        None
        """
        if self.kgrid is None:
            self._calculate_k_grid()

    def _calculate_k_grid(self):
        for i, ax in enumerate("xyz"):
            Nx, dx = self.grid_shape[i], self.dxyz[i]
            k = (2 * pi * np.fft.fftfreq(Nx, dx)).astype(config.FDTYPE)
            setattr(self, f"k{ax}", k)

        self.kgrid = (self.kx, self.ky, self.kz)
        self.kgrid_shifted = tuple(
            np.fft.fftshift(k) for k in (self.kx, self.ky, self.kz)
        )
        self.dkxkykz = [ax[1] - ax[0] for ax in self.kgrid]
        self.dVk = np.prod(self.dkxkykz)

        self.kmeshgrid = np.meshgrid(
            self.kx, self.ky, self.kz, indexing="ij", sparse=True
        )
        kx, ky, kz = self.kmeshgrid
        self.kabs = kabs = ne.evaluate("sqrt(kx**2 + ky**2 + kz**2)") # noqa: F841
        kperp = ne.evaluate("sqrt(kx**2 + ky**2)") # noqa: F841

        # Polarization vectors
        self.e1x = ne.evaluate("where((kx==0) & (ky==0), 1, kx * kz / (kperp*kabs))")
        self.e1y = ne.evaluate("where((kx==0) & (ky==0), 0, ky * kz / (kperp*kabs))")
        self.e1z = ne.evaluate("where((kx==0) & (ky==0), 0, -kperp / kabs)")

        self.e2x = ne.evaluate("where((kx==0) & (ky==0), 0, -ky / kperp)")
        self.e2y = ne.evaluate("where((kx==0) & (ky==0), 2*(kz>0)-1, kx / kperp)")
        self.e2z = 0


def get_ek(theta, phi):
    """
    Calculate k-vector for given spherical angles theta and phi.

    Parameters
    ----------
    theta : float
        The polar angle in radians.
    phi : float
        The azimuthal angle in radians.

    Returns
    -------
    numpy.ndarray
        A 3-element array representing the k-vector.
    """
    ek = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    return ek


def get_pol_basis(theta, phi):
    """
    Calculate polarization basis for given spherical angles theta and phi.

    Parameters
    ----------
    theta : float
        The polar angle in radians.
    phi : float
        The azimuthal angle in radians.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two 3-element arrays representing the polarization
        basis vectors e1 and e2.
    """
    e1 = np.array(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
        dtype=config.FDTYPE
    )
    e2 = np.array([-np.sin(phi), np.cos(phi), 0], dtype=config.FDTYPE)
    return e1, e2


def gaussian_bandwidth(field_params):
    """
    Calculate the gaussian bandwidth.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters. Required keys are:
            - tau : float
                Pulse duration.
            - w0 : float, optional
                Beam waist. If 'w0' is not provided, 'w0x' and 'w0y' are used.

    Returns
    -------
    tuple of float
        A tuple containing the perpendicular and longitudinal bandwidths 
        kbw_perp, kbw_long).

    Notes
    -----
    The bandwidths are calculated from the Fourier transform of transverse and 
    longitudinal Gaussian profiles:

    - exp(-r**2/w0**2) -> exp(-w0**2*kperp**2/4) -> kperp_bw ~ 2*2/w0
    
    - exp(-t**2/(tau/2)**2) -> exp(-tau**2*k**2/16) -> k_long ~ 2*4/(c*tau)
    """
    tau = field_params["tau"]

    # gaussian might have circular or elliptic cross section
    if "w0" in field_params:
        w0 = field_params["w0"]
    elif "w0x" in field_params:
        w0 = min(field_params["w0x"], field_params["w0y"])

    kbw_perp = 4 / w0
    kbw_long = 8 / (c * tau)

    return kbw_perp, kbw_long


def dipole_bandwidth(field_params):
    """
    Calculate the dipole bandwidth.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters. Required keys are:
            - lam : float
                Wavelength of the pulse.
            - tau : float
                Pulse duration.

    Returns
    -------
    tuple of float
        A tuple containing the perpendicular and longitudinal bandwidths 
        (kbw_perp, kbw_long).

    Notes
    -----
    The length scales are taken from:
    I. Gonoskov et al. "Dipole pulse theory: Maximizing the field amplitude 
    from 4π focused laser pulses." PRA 86.5 (2012): 053836.

    l_para = 0.58 * lam
    
    l_perp = 0.4 * lam
    """
    lam, tau = [field_params[k] for k in "lam tau".split()]
    
    l_long = 0.58*lam
    l_perp = 0.4*lam
    tau_bw = 8 / (c * tau)

    # Add up bandwidth coming from time and length scales
    kbw_perp = 4 / l_perp + tau_bw
    kbw_long = 4 / l_long + tau_bw

    return kbw_perp, kbw_long


def get_bw(field_params):
    """
    Calculate the bandwidths for a given field type.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters.

    Returns
    -------
    tuple of float
        A tuple containing the perpendicular and longitudinal bandwidths 
        (kbw_perp, kbw_long).

    Raises
    ------
    NotImplementedError
        If the field type is not supported.
    """
    ftype = field_params["field_type"]
    if "gaussian" in ftype:
        assert "w0" in field_params or "w0x" in field_params, (
            "Gaussian field parameters must have either 'w0' or 'w0x' key"
        )
        bws = gaussian_bandwidth(field_params)
    elif "dipole" in ftype:
        bws = dipole_bandwidth(field_params)
    return bws


def _check_keys(field_params, required_keys):
    err_msg = f"Field parameters must have {required_keys} keys"
    assert all(key in field_params for key in required_keys), err_msg


def get_kmax_from_bw(field_params):
    """
    Calculate the maximum k-vector from bandwidth along each dimension 
    for a given field.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters. Required keys are:
            - lam : float
                Wavelength of the pulse.
            - tau : float
                Pulse duration.
            - theta : float
                Polar angle in degrees.
            - phi : float
                Azimuthal angle in degrees.

    Returns
    -------
    numpy.ndarray
        A 3-element array representing the maximum k-vector along each dimension.

    Notes
    -----
    The maximum k-vector is calculated based on the bandwidths derived from the 
    field parameters.
    """
    required_keys = "lam tau theta phi".split()
    _check_keys(field_params, required_keys)

    lam, _, theta, phi = [field_params[k] for k in required_keys]

    k = 2 * np.pi / lam
    theta *= pi / 180
    phi = phi * pi / 180 if np.sin(theta) != 0.0 else 0.0

    ek = get_ek(theta, phi)
    e1, e2 = get_pol_basis(theta, phi)

    kbw_perp, kbw_long = get_bw(field_params)

    k0 = ek * k
    kmax = np.abs(k0 + ek * kbw_long)

    for beta in np.linspace(0, 2 * np.pi, 64, endpoint=False):
        kp = k0 + ek * kbw_long + kbw_perp * (np.cos(beta) * e1 + np.sin(beta) * e2)
        kmax = np.maximum(kmax, np.abs(kp))
    
    return kmax


def get_kmax_grid(field_params):
    """
    Calculate the maximum k-vector along each dimension for a given field type.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters.

    Returns
    -------
    numpy.ndarray
        A 3-element array representing the maximum k-vector along each dimension.

    Raises
    ------
    NotImplementedError
        If the field type is not supported.
    """
    ftype = field_params["field_type"]
    if ("gaussian" in ftype) or ("dipole" in ftype):
        kmax = get_kmax_from_bw(field_params)
    else:
        raise NotImplementedError(f"{ftype} field type is not supported")
    return kmax


def get_xyz_size(fields, box_size, grid_res=1, equal_resolution=False):
    """
    Calculate necessary spatial resolution.

    Parameters
    ----------
    fields : dict or list of dict
        Parameters of participating fields.
    box_size : sequence of float
        Box size for 3 dimensions.
    grid_res : float, optional
        Controls the resolution (default is 1).
    equal_resolution : bool, optional
        Flag indicating if equal resolution in every dimension is needed 
        (default is False).

    Returns
    -------
    list of int
        List containing the number of grid points along each dimension.
    """
    if isinstance(fields, dict):
        fields = list(fields.values())
    box_size = np.array(box_size)

    kmax = np.zeros((3,))
    for field_params in fields:
        kmax = np.maximum(kmax, get_kmax_grid(field_params))

    # if a square box is required
    if equal_resolution:
        kmax = np.max(kmax) * np.ones(3)

    N = grid_res * np.ceil(box_size * 3 * kmax / pi)
    N = [pyfftw.next_fast_len(int(n)) for n in N]
    return N


def get_t_size(t_start, t_end, lam, grid_res=1):
    """
    Calculate necessary temporal resolution.

    Parameters
    ----------
    t_start : float
        Start time.
    t_end : float
        End time.
    lam : float
        Wavelength.
    grid_res : float, optional
        Factor to scale the resolution (default is 1).

    Returns
    -------
    float
        Number of time steps.
    """
    fmax = c / lam
    return int(np.ceil((t_end - t_start) * fmax * 6 * grid_res))


def get_box_size(fields_params, grid_params):
    """
    Calculate the necessary box size for the spatial grid.

    Parameters
    ----------
    fields_params : list of dict
        Parameters of participating fields. Each dictionary should contain:
            - field_type : str
                Type of the field.
            - w0 : float, optional
                Beam waist for focused fields.
            - tau : float, optional
                Pulse duration.
    grid_params : dict
        Grid parameters. Required keys are:
            - transverse_factor : float
                Factor to scale the transverse size.
            - longitudinal_factor : float
                Factor to scale the longitudinal size.

    Returns
    -------
    tuple of float
        A tuple containing the transverse and longitudinal sizes.

    Notes
    -----
    The box size is calculated based on the maximum beam waist and pulse duration
    among the fields, scaled by the transverse and longitudinal factors.
    """
    perp_max = 0
    for field in fields_params:
        ftype = field["field_type"]
        if "gauss" in ftype:
            length = field.get("w0", 0)
        elif "dipole" in ftype:
            length = c * field.get("tau", 0) / 4
        else:
            length = 0
        perp_max = np.maximum(length, perp_max)
    
    tau_max = max([field.get("tau", 0) for field in fields_params])

    transverse_size = perp_max * grid_params["transverse_factor"]
    longitudinal_size = tau_max * c * grid_params["longitudinal_factor"]

    return float(transverse_size), float(longitudinal_size)


def create_dynamic_grid(fields_params, grid_params):
    """
    Dynamically create grids from given laser parameters.
    One should be careful with this option.

    Parameters
    ----------
    fields_params : list of dict
        Parameters of participating fields. 
    grid_params : dict
        Grid parameters.

    Returns
    -------
    dict
        Updated grid parameters including calculated box size, number of spatial points,
        and number of temporal points.
    """
    grid_params_upd = deepcopy(grid_params)
    # Create spatial box
    collision_geometry = grid_params.get("collision_geometry", "z")

    tau_max = max([field.get("tau", 0) for field in fields_params])
    lam_min = min([field.get("lam", 1e10) for field in fields_params])

    transverse_size, longitudinal_size = get_box_size(fields_params, grid_params)

    box_xyz = [0, 0, 0]
    for i, ax in enumerate("xyz"):
        box_xyz[i] = longitudinal_size if ax in collision_geometry else transverse_size
    grid_params_upd["box_xyz"] = box_xyz

    # Number of spatial pts
    box_size = np.array(box_xyz) / 2
    Nxyz_ = get_xyz_size(fields_params, box_size)
    res = grid_params.get("spatial_resolution", 1)
    if isinstance(res, Iterable):
        assert len(res) == 3, "Spatial resolution must be a list of 3 values or a "
        "single value"
        Nxyz = [Nx * res for Nx, res in zip(Nxyz_, res, strict=True)]
    elif type(res) in (int, float):
        Nxyz = [Nx * res for Nx in Nxyz_]
    grid_params_upd["Nxyz"] = Nxyz

    # Create temporal box
    t0 = tau_max * grid_params["time_factor"]
    Nt = get_t_size(-t0 / 2, t0 / 2, lam_min)

    # Number of temporal pts
    grid_params_upd["box_t"] = t0
    grid_params_upd["Nt"] = Nt * grid_params.get("time_resolution", 1)
    return grid_params_upd


def setup_grids(fields_params, grid_params):
    """
    Create spatial and temporal grids from given sizes or
    dynamically from field parameters (e.g., duration and waist size).

    Parameters
    ----------
    fields_params : list of dict
        Parameters of participating fields. Each dictionary should contain:
            - field_type : str
                Type of the field.
            - lam : float
                Wavelength of the pulse.
            - w0 : float, optional
                Beam waist for focused fields.
            - tau : float, optional
                Pulse duration.

    grid_params : dict
        Grid parameters. Required keys are:
            -  mode : str
                Mode of grid creation ('dynamic' or 'static').
        Keys for 'static' mode:
            - box_xyz : tuple of float
                Box size for the spatial grid.
            - Nxyz : tuple of int
                Number of grid points along each spatial dimension.
            - Nt : int
                Number of temporal points.
            - box_t : float or tuple of float
                Time duration or start and end times for the temporal grid.
        Keys for 'dynamic' mode:
            - collision_geometry : str
                Specifies the collision geometry ('x', 'y', 'z').
            - transverse_factor : float
                Factor to scale the transverse size.
            - longitudinal_factor : float
                Factor to scale the longitudinal size.
            - time_factor : float
                Factor to scale the time duration.
            - spatial_resolution : float or list of float, optional
                Controls the spatial resolution.
            - time_resolution : float, optional
                Controls the temporal resolution.
            - ignore_idx : list of int, optional
                Indices of fields to ignore for dynamic grid creation.

    Returns
    -------
    tuple
        A tuple containing:
            - grid_xyz : quvac.grid.GridXYZ
                The spatial grid object.
            - grid_t : numpy.ndarray
                The temporal grid array.

    Notes
    -----
    The spatial and temporal grids are created based on the mode specified in the grid 
    parameters. If 'dynamic' mode is selected, the grids are created dynamically based 
    on the field parameters.
    """
    if grid_params["mode"] == "dynamic":
        if isinstance(fields_params, dict):
            fields_params = list(fields_params.values())
        # filter out fields that should not contribute to dynamic grid creation
        ignore_idx = grid_params.get("ignore_idx", None)
        if ignore_idx is not None:
            fields_params = [field for idx,field in enumerate(fields_params) 
                            if idx not in ignore_idx]
        grid_params = create_dynamic_grid(fields_params, grid_params)

    x0, y0, z0 = grid_params["box_xyz"]
    Nx, Ny, Nz = grid_params["Nxyz"]
    x = np.linspace(-0.5 * x0, 0.5 * x0, Nx, endpoint=Nx % 2, dtype=config.FDTYPE)
    y = np.linspace(-0.5 * y0, 0.5 * y0, Ny, endpoint=Ny % 2, dtype=config.FDTYPE)
    z = np.linspace(-0.5 * z0, 0.5 * z0, Nz, endpoint=Nz % 2, dtype=config.FDTYPE)
    grid_xyz = GridXYZ((x, y, z))

    Nt = grid_params["Nt"]
    box_t = grid_params["box_t"]
    # Allow non-symmetric time-grids
    if isinstance(box_t, float):
        Nt += int(1 - Nt % 2)
        t0 = box_t
        grid_t = np.linspace(
            -0.5 * t0, 0.5 * t0, Nt, endpoint=True, dtype=config.FDTYPE
        )
    else:
        t_start, t_end = box_t
        grid_t = np.linspace(t_start, t_end, Nt, endpoint=True, dtype=config.FDTYPE)
    return grid_xyz, grid_t
