"""
Here we provide analyzer classes that calculate from amplitudes:
    - Total (polarization insensitive) signal
    - Polarization sensitive signal
    - Discernible signal
"""

import logging
import os
import warnings

import numexpr as ne
import numpy as np
from scipy.constants import c, epsilon_0, hbar, pi
from scipy.integrate import trapezoid
from scipy.ndimage import map_coordinates

from quvac import config
from quvac.field.maxwell import MaxwellMultiple
from quvac.field.external_field import ExternalField
from quvac.grid import GridXYZ, get_pol_basis, setup_grids
from quvac.log import sph_interp_warn
from quvac.utils import read_yaml

logger = logging.getLogger("simulation")


def get_polarization_vector(theta, phi, beta):
    e1, e2 = get_pol_basis(theta, phi)
    ep = e1 * np.cos(beta, dtype=config.FDTYPE) + e2 * np.sin(beta, dtype=config.FDTYPE)
    return ep


def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta) * np.ones_like(phi)
    return x, y, z


def xyz2idx(xyz, xyz_grid):
    nx, ny, nz = xyz[0].shape
    idxs = np.empty((3, nx, ny, nz))
    for i, (x, grid) in enumerate(zip(xyz, xyz_grid)):
        x0, x1 = grid[0], grid[-1]
        idxs[i] = (x - x0) / (x1 - x0) * (len(grid) - 1)
    return idxs


def cartesian_to_spherical_array(
    arr, xyz_grid, spherical_grid=None, angular_resolution=None, **interp_kwargs
):
    """
    Transforms an array with data on cartesian grid to the
    array with data on spherical grid
    """
    # Calculate spherical grid if not given
    if not spherical_grid:
        dk = np.min(xyz_grid.dkxkykz)
        kmax = np.max(xyz_grid.kabs)
        dangle = angular_resolution if angular_resolution else 1.0 * pi / 180

        k = np.arange(0.0, kmax, dk, dtype=config.FDTYPE)
        theta = np.arange(0.0, pi, dangle, dtype=config.FDTYPE)
        phi = np.arange(0.0, 2 * pi, dangle, dtype=config.FDTYPE)
        spherical_grid = (k, theta, phi)
    elif isinstance(spherical_grid, str) and os.path.isfile(spherical_grid):
        data = np.load(spherical_grid)
        spherical_grid = (data["k"], data["theta"], data["phi"])
    spherical_mesh = np.meshgrid(*spherical_grid, indexing="ij", sparse=True)

    # Find corresponding cartesian coordinates of spherical mesh:
    # (r,theta,phi) -> (x, y, z)
    xyz_for_sph = sph2cart(*spherical_mesh)

    # Convert cartesian coordinates to array idx
    idxs = xyz2idx(xyz_for_sph, xyz_grid.kgrid_shifted)
    # idxs = np.stack(idxs, axis=0)

    # Interpolate data on a desired grid
    arr_sph = map_coordinates(arr, idxs, order=1)
    return spherical_grid, arr_sph


def integrate_spherical(arr, axs, axs_names=["k", "theta", "phi"],
                        axs_integrate=["k", "theta", "phi"]):
    err_msg = "Axes of array and axs names do not match"
    assert len(axs) == len(axs_names) and len(axs) >= len(axs_integrate), err_msg

    axs_names_ = axs_names.copy()

    integrand = arr.copy()
    for ax_name in axs_integrate:
        idx = axs_names.index(ax_name)
        idx_ = axs_names_.index(ax_name)

        ax_shape = [1 for _ in range(len(axs_names_))]
        ax_shape[idx_] = integrand.shape[idx_]

        ax = axs[idx].reshape(ax_shape)
        if ax_name == "k":
            integrand *= ax**2
        elif ax_name == "theta":
            integrand *= np.sin(ax)
        integrand = trapezoid(integrand, axs[idx], axis=idx_)
        axs_names_.pop(idx_)
    return integrand


def get_simulation_fields(ini_file):
    ini = read_yaml(ini_file)
    fields_params = ini["fields"]
    grid_params = ini["grid"]

    grid_xyz, grid_t = setup_grids(fields_params, grid_params)
    grid_xyz.get_k_grid()

    fields = []
    for field_params in fields_params.values():
        field = ExternalField([field_params], grid_xyz)
        fields.append(field)
    return fields


def get_a12_from_amplitudes(S1, S2, k):
    '''
    Given complex signal amplitudes S1 and S2, calculate
    spectral coefficients a1 and a2 (similar to Maxwell fields)
    k = kabs
    '''
    prefactor = np.sqrt(2*hbar*k/(epsilon_0*c))
    a1 = S1 * prefactor
    a2 = S2 * prefactor
    return a1, a2


class VacuumEmissionAnalyzer:
    """
    Calculates spectra and observables from amplitudes
    provided by quvac.integrator.vacuum_emission.VacuumEmission
    class

    Currently supports:
        - Differential polarization-(in)sensitive spectrum on (kx,ky,kz) grid
        - Differential polarization-(in)sensitive spectrum on (k,theta,phi) grid
        - Total signal
    """

    def __init__(self, fields_params, data_path, save_path=None):
        self.fields_params = fields_params
        # Load data
        data = np.load(data_path)
        grid = tuple((data["x"], data["y"], data["z"]))
        self.grid_xyz = GridXYZ(grid)
        self.grid_xyz.get_k_grid()
        # Update local dict with variables from GridXYZ class
        self.__dict__.update(self.grid_xyz.__dict__)

        for ax in "xyz":
            self.__dict__[f"k{ax}"] = np.fft.fftshift(self.__dict__[f"k{ax}"])

        self.S1, self.S2 = data["S1"].astype(config.CDTYPE), data["S2"].astype(config.CDTYPE)

        self.save_path = save_path

    def get_total_signal(self):
        S = self.S1.real**2 + self.S1.imag**2 + self.S2.real**2 + self.S2.imag**2
        self.N_xyz = np.fft.fftshift(S / (2 * pi) ** 3)

        self.N_total = np.sum(self.N_xyz) * self.dVk

    def get_polarization_from_field(self):
        perp_field_params = self.fields_params[self.perp_field_idx]
        field = MaxwellMultiple([perp_field_params], self.grid_xyz)
        # field = MaxwellMultiple(self.fields_params, self.grid_xyz)
        a1, a2 = field.a1, field.a2
        Ex = ne.evaluate("e1x*a1 + e2x*a2", global_dict=self.__dict__)
        Ey = ne.evaluate("e1y*a1 + e2y*a2", global_dict=self.__dict__)
        Ez = ne.evaluate("e1z*a1 + e2z*a2", global_dict=self.__dict__)
        E = ne.evaluate("sqrt(Ex**2 + Ey**2 + Ez**2)")
        # E_inv = np.where(E/E.max() > 1e-15, np.nan_to_num(1. / E), 0.)
        E_inv = np.nan_to_num(1. / E)
        efx, efy, efz = Ex * E_inv, Ey * E_inv, Ez * E_inv
        return (efx, efy, efz)

    # def _get_polarization_vector(self, angles, perp_type="optical axis"):
    #     if perp_type == "optical axis":
    #         self.epx, self.epy, self.epz = get_polarization_vector(*angles)
    #     elif perp_type == "local axis":
    #         efx, efy, efz = self.get_polarization_from_field()
    #         kx, ky, kz = [k / self.kabs for k in self.kmeshgrid]
    #         kx[0, 0, 0] = 0.0
    #         ky[0, 0, 0] = 0.0
    #         kz[0, 0, 0] = 0.0
    #         self.epx = ne.evaluate("ky*efz - kz*efy")
    #         self.epy = ne.evaluate("kz*efx - kx*efz")
    #         self.epz = ne.evaluate("kx*efy - ky*efx")
    #     return (self.epx, self.epy, self.epz)

    def _get_polarization_vector(self, angles, perp_type="optical axis"):
        '''
        Calculate polarization vectors parallel to chosen axis and perp to it
        ef - field polarization (parallel)
        ep - perp polarization
        '''
        if perp_type == "optical axis":
            self.efx, self.efy, self.efz = get_polarization_vector(*angles)
            angles[-1] += pi / 2
            self.epx, self.epy, self.epz = get_polarization_vector(*angles)
        elif perp_type == "local axis":
            efx, efy, efz = self.get_polarization_from_field()
            self.efx, self.efy, self.efz = efx, efy, efz
            kx, ky, kz = [k / self.kabs for k in self.kmeshgrid]
            kx[0, 0, 0] = 0.0
            ky[0, 0, 0] = 0.0
            kz[0, 0, 0] = 0.0
            self.epx = ne.evaluate("ky*efz - kz*efy")
            self.epy = ne.evaluate("kz*efx - kx*efz")
            self.epz = ne.evaluate("kx*efy - ky*efx")
        return (self.efx, self.efy, self.efz), (self.epx, self.epy, self.epz)

    def get_perp_signal(self, angles, perp_type="optical axis",
                        stokes=False):
        """
        angles (theta, phi, beta): (float, float, float)
            Euler angles for field polarization (in degrees)
        """
        angles = [angle * pi / 180 for angle in angles]
        # Here we make sure that perp polarization would be calculated
        # angles[-1] += pi / 2

        # get one polarization direction to project on:
        # self.ep = self._get_polarization_vector(angles, perp_type)
        self.ef, self.ep = self._get_polarization_vector(angles, perp_type)
        epx, epy, epz = self.ep
        e1x, e1y, e1z = self.e1x, self.e1y, self.e1z
        e2x, e2y, e2z = self.e2x, self.e2y, self.e2z

        # Calculate perp signal or stokes parameters
        if stokes:
            self.get_Stokes_vector_for_ep()
        else:
            Sp = (epx*e1x + epy*e1y + epz*e1z)*self.S1 + (epx*e2x + epy*e2y + epz*e2z)*self.S2
            Sp = Sp.real**2 + Sp.imag**2

            self.Np_xyz = np.fft.fftshift(Sp / (2 * pi) ** 3)
            self.Np_total = np.sum(self.Np_xyz) * self.dVk

    def get_Stokes_vector_for_ep(self):
        '''
        Given amplitudes S1,S2 for arbitrary linear polarization
        basis e1,e2, calculate Stokes vectors for detector ep
        '''
        e1x, e1y, e1z = self.e1x, self.e1y, self.e1z
        e2x, e2y, e2z = self.e2x, self.e2y, self.e2z
        efx, efy, efz = self.ef
        epx, epy, epz = self.ep

        Sf = (efx*e1x + efy*e1y + efz*e1z)*self.S1 + (efx*e2x + efy*e2y + efz*e2z)*self.S2
        Sp = (epx*e1x + epy*e1y + epz*e1z)*self.S1 + (epx*e2x + epy*e2y + epz*e2z)*self.S2

        P1 = Sf.real**2 + Sf.imag**2 - (Sp.real**2 + Sp.imag**2)
        P2 = 2 * np.real(Sf * np.conj(Sp))
        P3 = -2 * np.imag(Sf * np.conj(Sp))

        self.P1, self.P2, self.P3 = [np.fft.fftshift(P / (2 * pi) ** 3)
                                     for P in (P1,P2,P3)]

    def get_signal_on_sph_grid(
        self, key="N_xyz", spherical_grid=None, angular_resolution=None, **interp_kwargs
    ):
        arr = getattr(self, key)
        spherical_grid, N_sph = cartesian_to_spherical_array(
            arr,
            self.grid_xyz,
            spherical_grid=spherical_grid,
            angular_resolution=angular_resolution,
            **interp_kwargs,
        )
        sph_key = key.replace("xyz", "sph")
        sph_total_key = f"{sph_key}_total"
        total_key = key.replace("xyz", "total")
        self.__dict__[sph_key] = N_sph

        N_total = integrate_spherical(N_sph, spherical_grid)
        self.__dict__[sph_total_key] = N_total

        if not np.isclose(self.__dict__[total_key], N_total, rtol=1e-2):
            warn_message = sph_interp_warn.format(
                total_key, self.__dict__[total_key], N_total
            )
            warnings.warn(warn_message)
            logger.warning(warn_message)

        self.spherical_grid = self.k, self.theta, self.phi = spherical_grid

    def get_photon_spectrum_from_a12(self, a1, a2):
        prefactor = 0.5 * epsilon_0 * c / hbar
        k = self.kabs
        N_xyz = ne.evaluate(
            "prefactor * (a1.real**2 + a1.imag**2 + a2.real**2 + a2.imag**2) / k"
        )
        N_xyz[0, 0, 0] = 0.0
        return np.fft.fftshift(N_xyz)
    
    def get_background_xyz(self, add_to_cls_dict=True, bgr_idx=None):
        if bgr_idx is not None:
            bgr_field = MaxwellMultiple(self.fields_params[bgr_idx], self.grid_xyz)
        else:
            bgr_field = MaxwellMultiple(self.fields_params, self.grid_xyz)
        bgr_N_xyz = self.get_photon_spectrum_from_a12(bgr_field.a1, bgr_field.a2)
        if add_to_cls_dict:
            self.background_xyz = bgr_N_xyz
        return bgr_N_xyz

    def get_background(self, discernibility="angular", bgr_idx=None,
                       **interp_kwargs):
        bgr_N_xyz = self.get_background_xyz(add_to_cls_dict=False, bgr_idx=bgr_idx)

        # Interpolate on spherical grid
        _, bgr_N_sph = cartesian_to_spherical_array(
            bgr_N_xyz,
            self.grid_xyz,
            spherical_grid=self.__dict__.get('spherical_grid', None),
            **interp_kwargs,
        )

        # Integrate over k if needed
        if discernibility == "angular":
            bgr_N_sph = integrate_spherical(
                bgr_N_sph, self.spherical_grid, axs_integrate=["k"]
            )
        return bgr_N_sph

    def get_discernible_signal(self, discernibility="angular"):
        # Calculate numerical background
        self.background = self.get_background(discernibility=discernibility)

        # Integrate signal spectrum if required and determine discernible regions
        if discernibility == "angular":
            self.N_angular = integrate_spherical(
                self.N_sph, self.spherical_grid, axs_integrate=["k"]
            )
            self.discernible = self.N_angular > self.background
        else:
            self.discernible = self.N_sph > self.background

        # Integrate over discernible regions
        self.N_disc = integrate_spherical(
            self.N_sph * self.discernible, self.spherical_grid
        )
    
    def get_signal_a12(self, add_signal_bg=False):
        self.a1_sig, self.a2_sig = get_a12_from_amplitudes(self.S1, self.S2, self.kabs)
        if add_signal_bg:
            bgr_field = MaxwellMultiple(self.fields_params, self.grid_xyz)
            a1_bgr, a2_bgr = bgr_field.a1, bgr_field.a2
            self.a1_mix, self.a2_mix = self.a1_sig + a1_bgr, self.a2_sig + a2_bgr
        self.a1_sig, self.a2_sig = [np.fft.fftshift(a) for a 
                                    in (self.a1_sig,self.a2_sig)]
        self.a1_mix, self.a2_mix = [np.fft.fftshift(a) for a 
                                    in (self.a1_mix,self.a2_mix)]

    def write_data(self, keys):
        data = {key: getattr(self, key) for key in keys}
        np.savez(self.save_path, **data)

    def get_total_spectra(
        self,
        calculate_xyz_background=False,
        bgr_idx=None,
        calculate_spherical=False,
        spherical_params=None,
        calculate_discernible=False,
        discernibility="angular",
    ):
        self.get_total_signal()
        keys = "kx ky kz N_xyz N_total".split()

        if calculate_spherical:
            self.get_signal_on_sph_grid(key="N_xyz", **spherical_params)
            keys.extend("k theta phi N_sph N_sph_total".split())
        if calculate_xyz_background:
            self.get_background_xyz()
            self.background = self.get_background(discernibility=None, bgr_idx=bgr_idx)
            keys.extend("background_xyz background".split())
        if calculate_discernible:
            self.get_discernible_signal(discernibility)
            keys.extend("background discernible N_disc".split())
        
        self.write_data(keys)

    def get_polarization_spectra(
        self,
        perp_field_idx=1,
        perp_type=None,
        calculate_spherical=False,
        spherical_params=None,
        stokes=False,
    ):
        self.perp_field_idx = perp_field_idx - 1
        angle_keys = "theta phi beta".split()
        angles = [self.fields_params[self.perp_field_idx][key] for key in angle_keys]
        self.get_perp_signal(angles, perp_type=perp_type, stokes=stokes)
        # keys = "kx ky kz Np_xyz Np_total epx epy epz".split()
        keys = "kx ky kz epx epy epz efx efy efz".split()

        if not stokes:
            keys.extend("Np_xyz Np_total".split())
            if calculate_spherical:
                self.get_signal_on_sph_grid(key="Np_xyz", **spherical_params)
                keys.extend("k theta phi Np_sph Np_sph_total".split())
        else:
            stokes_keys = "P1 P2 P3".split()
            keys.extend(stokes_keys)
            if calculate_spherical:
                keys.extend("k theta phi".split())
                for stokes_key in stokes_keys:
                    self.get_signal_on_sph_grid(key=stokes_key, **spherical_params)
                    keys.extend([f"{stokes_key}_sph"])
        
        self.write_data(keys)

    def get_mix_signal_bg(self, add_signal_bg=False):
        self.get_signal_a12(add_signal_bg)

        keys = "a1_sig a2_sig".split()
        if add_signal_bg:
            keys.extend("a1_mix a2_mix".split())
        self.write_data(keys)

    def get_spectra(
        self,
        mode="total",
        perp_field_idx=1,
        perp_type=None,
        calculate_xyz_background=False,
        bgr_idx=None,
        stokes=False,
        calculate_spherical=False,
        spherical_params=None,
        calculate_discernible=False,
        discernibility="angular",
        add_signal_bg=False,
    ):
        if mode == "total":
            self.get_total_spectra(
                calculate_xyz_background=calculate_xyz_background,
                bgr_idx=bgr_idx,
                calculate_spherical=calculate_spherical,
                spherical_params=spherical_params,
                calculate_discernible=calculate_discernible,
                discernibility=discernibility
            )
        elif mode == "polarization":
            self.get_polarization_spectra(
                perp_field_idx=perp_field_idx,
                perp_type=perp_type,
                calculate_spherical=calculate_spherical,
                spherical_params=spherical_params,
                stokes=stokes,
            )
        elif mode == "mix_signal_bg":
            self.get_mix_signal_bg(add_signal_bg=add_signal_bg)

