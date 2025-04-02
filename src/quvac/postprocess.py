"""
Analyzer class that calculate different observables from amplitudes.

Currently supports:

1. Total (polarization insensitive) signal.
2. Polarization sensitive signal:
    - As a projection to fixed optical axis.
    - As a projection to local polarization axis.
    - 3 stokes parameters (P1, P2, P3) for arbitrary 
      linear polarization basis.

3. Discernible signal.
4. Background field spectra.
"""
# ruff: noqa: F841
import logging
import os
import warnings

import numexpr as ne
import numpy as np
from scipy.constants import c, epsilon_0, hbar, pi
from scipy.integrate import trapezoid
from scipy.ndimage import map_coordinates

from quvac import config
from quvac.field.external_field import ExternalField
from quvac.field.maxwell import MaxwellMultiple
from quvac.grid import GridXYZ, get_pol_basis, setup_grids
from quvac.log import sph_interp_warn
from quvac.utils import read_yaml

_logger = logging.getLogger("simulation")


def get_polarization_vector(theta, phi, beta):
    """
    Calculate the polarization vector.

    Parameters
    ----------
    theta : float
        Polar angle (in radians).
    phi : float
        Azimuthal angle (in radians).
    beta : float
        Polarization angle (in radians).

    Returns
    -------
    ep : numpy.ndarray
        Polarization vector.
    """
    e1, e2 = get_pol_basis(theta, phi)
    ep =(e1 * np.cos(beta, dtype=config.FDTYPE) + 
         e2 * np.sin(beta, dtype=config.FDTYPE))
    return ep


def sph2cart(r, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : numpy.ndarray
        Radial distance.
    theta : numpy.ndarray
        Polar angle.
    phi : numpy.ndarray
        Azimuthal angle.

    Returns
    -------
    x, y, z : numpy.ndarray
        Cartesian coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta) * np.ones_like(phi)
    return x, y, z


def xyz2idx(xyz, xyz_grid):
    """
    Convert Cartesian coordinates to array indices.

    Parameters
    ----------
    xyz : tuple of numpy.ndarray
        Cartesian coordinates.
    xyz_grid : tuple of numpy.ndarray
        Cartesian grid.

    Returns
    -------
    idxs : numpy.ndarray
        Array indices.
    """
    nx, ny, nz = xyz[0].shape
    idxs = np.empty((3, nx, ny, nz))
    for i, (x, grid) in enumerate(zip(xyz, xyz_grid)):
        x0, x1 = grid[0], grid[-1]
        idxs[i] = (x - x0) / (x1 - x0) * (len(grid) - 1)
    return idxs


def cartesian_to_spherical_array(
    arr, xyz_grid, spherical_grid=None, angular_resolution=None,
    **interp_kwargs
):
    """
    Transform an array with data on Cartesian grid to the array with data 
    on spherical grid.

    Parameters
    ----------
    arr : numpy.ndarray
        Array with data on Cartesian grid.
    xyz_grid : quvac.grid.GridXYZ
        Cartesian grid object.
    spherical_grid : tuple or str, optional
        Spherical grid or path to the file containing the spherical grid, 
        by default None.
    angular_resolution : float, optional
        Angular resolution, by default None.
    **interp_kwargs : dict
        Additional interpolation parameters. Currently not used, implement 
        in the future.

    Returns
    -------
    spherical_grid : tuple
        Spherical grid.
    arr_sph : numpy.ndarray
        Array with data on spherical grid.
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

    # Interpolate data on a desired grid
    # interpolation_kwargs should be implemented here
    arr_sph = map_coordinates(arr, idxs, order=1)
    return spherical_grid, arr_sph


def integrate_spherical(arr, axs, axs_names=["k", "theta", "phi"],
                        axs_integrate=["k", "theta", "phi"]):
    """
    Integrate an array over spherical coordinates.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be integrated.
    axs : list of numpy.ndarray
        List of axes.
    axs_names : list of str, optional
        List of axis names, by default ["k", "theta", "phi"].
    axs_integrate : list of str, optional
        List of axes to integrate over, by default ["k", "theta", "phi"].

    Returns
    -------
    integrand : numpy.ndarray
        Integrated array.

    Raises
    ------
    AssertionError
        Axes of array and axs names do not match.

    Notes
    -----
    We assume that the array is given on a spherical grid without
    the jacobian factor, therefore the array is mutiplied by k**2 sin(theta).
    
    The function uses trapezoidal rule for integration.
    """
    err_msg = "Axes of array and axs names do not match"
    condition = len(axs) == len(axs_names) and len(axs) >= len(axs_integrate)
    assert condition, err_msg

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


def _get_detector_idx(phi, theta, phi0, theta0, dphi, dtheta):
    # consider detector regions that lie on the line phi=0 or phi=2*pi
    if phi0-dphi < 0:
        idx_phi = (phi <= phi0+dphi) + (phi >= 2*pi-abs(phi0-dphi))
    elif phi0+dphi > 2*pi:
        idx_phi = (phi >= phi0-dphi) + (phi <= phi0+dphi-2*pi)
    else:
        idx_phi = (phi >= phi0-dphi) * (phi <= phi0+dphi)
    
    idx_theta = (theta >= theta0-dtheta) * (theta <= theta0+dtheta)
    return idx_phi, idx_theta


def signal_in_detector(dN, theta, phi, detector, align_to_max=False):
    """
    Calculate the signal detected within a specified detector region.

    Parameters
    ----------
    dN : numpy.ndarray
        Differential angular photon spectrum on a spherical grid.
    theta : numpy.ndarray
        Array of polar angles (in radians) corresponding to the spherical grid.
    phi : numpy.ndarray
        Array of azimuthal angles (in radians) corresponding to the spherical grid.
    detector : dict
        Dictionary specifying the detector region. Required keys are:
        - 'phi0' : float
            Central azimuthal angle of the detector (in degrees).
        - 'theta0' : float
            Central polar angle of the detector (in degrees).
        - 'dphi' : float
            Half-width of the azimuthal angle range (in degrees).
        - 'dtheta' : float
            Half-width of the polar angle range (in degrees).
    align_to_max : bool
        Whether to align detector to the max spot in the detected region, 
        by default False.

    Returns
    -------
    float
        The total signal detected within the specified detector region.

    Notes
    -----
    The function integrates the signal over the specified detector region in spherical 
    coordinates.
    """
    phi0, theta0, dphi0, dtheta0 = [np.radians(detector[key]) for key 
                                  in "phi0 theta0 dphi dtheta".split()]
    idx_phi,idx_theta = _get_detector_idx(phi, theta, phi0, theta0, dphi0, dtheta0)
    dphi, dtheta = [ax[1]-ax[0] for ax in (phi, theta)]
    
    dN_det = dN[idx_theta][:,idx_phi]
    theta_det, phi_det = theta[idx_theta], phi[idx_phi]

    if align_to_max:
        max_index = np.argmax(dN_det)
        theta_max, phi_max = np.unravel_index(max_index, dN_det.shape)
        phi0_new, theta0_new = phi_det[phi_max], theta_det[theta_max]
        idx_phi,idx_theta = _get_detector_idx(phi, theta, phi0_new, theta0_new,
                                              dphi0, dtheta0)
        dN_det = dN[idx_theta][:,idx_phi]
        theta_det, phi_det = theta[idx_theta], phi[idx_phi]

    N_detected = np.sum(dN_det * np.sin(theta_det)[:,None]) * dphi * dtheta
    # N_detected = integrate_spherical(dN_det, [theta_det, phi_det],
    #                                  axs_names=['theta','phi'],
    #                                  axs_integrate=['theta','phi'])
    return N_detected
    

def get_simulation_fields(ini_file):
    """
    Get simulation fields from an initialization file.

    Parameters
    ----------
    ini_file : str
        Path to the initialization file.

    Returns
    -------
    fields : list of quvac.field.ExternalField
        List of field objects.
    """
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


def transform_S12_to_a12(S1, S2, k, transform="forward"):
    """
    Calculate spectral coefficients a1 and a2 from complex signal
    amplitudes S1 and S2.

    Parameters
    ----------
    S1 : numpy.ndarray
        Complex signal amplitude S1.
    S2 : numpy.ndarray
        Complex signal amplitude S2.
    k : numpy.ndarray
        Wave number.
    transform : str, optional
        Transformation direction, by default "forward".

    Returns
    -------
    a1, a2 : numpy.ndarray
        Spectral coefficients a1 and a2.

    Notes
    -----
    The backward transform corresponds to a12 -> S12.
    """
    prefactor = np.sqrt(2*hbar*k/(epsilon_0*c))
    if transform == "backward":
        prefactor = np.where(prefactor != 0, 1 / prefactor, 0.)
    a1 = S1 * prefactor
    a2 = S2 * prefactor
    return a1, a2


def get_spectra_from_Stokes(P0, P1, P2, P3, basis="linear"):
    """
    Calculate spectra from Stokes parameters.

    Parameters
    ----------
    P0 : numpy.ndarray
        Stokes parameter P0.
    P1 : numpy.ndarray
        Stokes parameter P1.
    P2 : numpy.ndarray
        Stokes parameter P2.
    P3 : numpy.ndarray
        Stokes parameter P3.
    basis : str, optional
        Polarization basis, by default "linear".

    Returns
    -------
    Nf, Np : numpy.ndarray
        Parallel and perpendicular-polarized spectra.
    """
    match basis:
        case "linear":
            P = P1
        case "linear-45":
            P = P2
        case "circular":
            P = P3
    Nf = 0.5 * (P0 + P)
    Np = 0.5 * (P0 - P)
    return Nf, Np


class VacuumEmissionAnalyzer:
    """
    Calculates spectra and observables from amplitudes.

    Amplitudes are provided by 
    ``quvac.integrator.vacuum_emission.VacuumEmission`` class.

    Parameters
    ----------
    fields_params : dict
        Dictionary containing the field parameters.
    data_path : str
        Path to the data file.
    save_path : str, optional
        Path to save the results, by default None.

    Attributes
    ----------
    fields_params : dict
        Dictionary containing the field parameters.
    grid_xyz : quvac.grid.GridXYZ
        The spatial grid object.
    S1 : numpy.ndarray
        Complex signal amplitude S1.
    S2 : numpy.ndarray
        Complex signal amplitude S2.
    save_path : str
        Path to save the results.
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
            kx = getattr(self, f"k{ax}")
            setattr(self, f"k{ax}", np.fft.fftshift(kx))
            # self.__dict__[f"k{ax}"] = np.fft.fftshift(kx)

        self.S1, self.S2 = (data["S1"].astype(config.CDTYPE),
                            data["S2"].astype(config.CDTYPE))

        self.save_path = save_path

    def get_total_signal(self):
        """
        Calculates the total signal.
        """
        S = self.S1.real**2 + self.S1.imag**2 + self.S2.real**2 + self.S2.imag**2
        self.N_xyz = np.fft.fftshift(S / (2 * pi) ** 3)

        self.N_total = np.sum(self.N_xyz) * self.dVk

    def get_polarization_from_field(self):
        """
        Calculates the polarization from the field.

        Returns
        -------
        tuple
            Tuple containing the polarization components (efx, efy, efz).
        """
        perp_field_params = self.fields_params[self.perp_field_idx]
        field = MaxwellMultiple([perp_field_params], self.grid_xyz)
        a1, a2 = field.a1, field.a2
        Ex = ne.evaluate("e1x*a1 + e2x*a2", global_dict=self.__dict__)
        Ey = ne.evaluate("e1y*a1 + e2y*a2", global_dict=self.__dict__)
        Ez = ne.evaluate("e1z*a1 + e2z*a2", global_dict=self.__dict__)
        E = ne.evaluate("sqrt(Ex**2 + Ey**2 + Ez**2)")
        E_inv = np.nan_to_num(1. / E)
        efx, efy, efz = Ex * E_inv, Ey * E_inv, Ez * E_inv
        return (efx, efy, efz)

    def _get_polarization_vector(self, angles, perp_type="optical axis"):
        """
        Calculates polarization vectors parallel to chosen axis and 
        perpendicular to it.

        Parameters
        ----------
        angles : tuple
            Tuple containing the angles (theta, phi, beta).
        perp_type : str, optional
            Type of perpendicular polarization, by default "optical axis".

        Returns
        -------
        tuple
            Tuple containing the parallel and perpendicular polarization 
            components.
        """
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
        else:
            raise NotImplementedError(f"{perp_type} is not implemented")
        return (self.efx, self.efy, self.efz), (self.epx, self.epy, self.epz)

    def get_perp_signal(self, angles, perp_type="optical axis",
                        stokes=False):
        """
        Calculates the perpendicular signal.

        Parameters
        ----------
        angles : tuple
            Tuple containing the angles (theta, phi, beta) in degrees.
        perp_type : str, optional
            Type of perpendicular polarization, by default "optical axis".
        stokes : bool, optional
            Whether to calculate Stokes parameters, by default False.
        """
        angles = [angle * pi / 180 for angle in angles]

        # get one polarization direction to project on:
        self.ef, self.ep = self._get_polarization_vector(angles, perp_type)
        epx, epy, epz = self.ep
        e1x, e1y, e1z = self.e1x, self.e1y, self.e1z
        e2x, e2y, e2z = self.e2x, self.e2y, self.e2z

        # Calculate perp signal or stokes parameters
        if stokes:
            self.get_Stokes_vector_for_ep()
        else:
            Sp = ((epx*e1x + epy*e1y + epz*e1z)*self.S1 + 
                  (epx*e2x + epy*e2y + epz*e2z)*self.S2)
            Sp = Sp.real**2 + Sp.imag**2

            self.Np_xyz = np.fft.fftshift(Sp / (2 * pi) ** 3)
            self.Np_total = np.sum(self.Np_xyz) * self.dVk

    def get_Stokes_vector_for_ep(self):
        '''
        Given amplitudes S1,S2 for arbitrary linear polarization
        basis e1,e2, calculate Stokes vectors for detector ep.

        For given ef (||) and ep (perp):
            - P0 = N_xyz = (S_||)**2 + (S_perp)**2

            - P1 = (S_||)**2 - (S_perp)**2
            
            - P2 = (S_45)**2 - (S_135)**2     - difference of linear 
              polarizations in 45 basis.
            
            - P3 = (S_right)**2 - (S_left)**2 - difference of circular 
              polarizations.

        '''
        e1x, e1y, e1z = self.e1x, self.e1y, self.e1z
        e2x, e2y, e2z = self.e2x, self.e2y, self.e2z
        efx, efy, efz = self.ef
        epx, epy, epz = self.ep

        Sf = ((efx*e1x + efy*e1y + efz*e1z)*self.S1 + 
              (efx*e2x + efy*e2y + efz*e2z)*self.S2)
        Sp = ((epx*e1x + epy*e1y + epz*e1z)*self.S1 + 
              (epx*e2x + epy*e2y + epz*e2z)*self.S2)

        P1 = Sf.real**2 + Sf.imag**2 - (Sp.real**2 + Sp.imag**2)
        P2 = 2 * np.real(Sf * np.conj(Sp))
        P3 = -2 * np.imag(Sf * np.conj(Sp))

        self.P1, self.P2, self.P3 = [np.fft.fftshift(P / (2 * pi) ** 3)
                                     for P in (P1,P2,P3)]

    def get_signal_on_sph_grid(
        self, key="N_xyz", spherical_grid=None, angular_resolution=None,
        check_total=False, **interp_kwargs
    ):
        """
        Transforms an array with data on Cartesian grid to the array with 
        data on spherical grid.

        Parameters
        ----------
        key : str, optional
            Key for the data array, by default "N_xyz".
        spherical_grid : tuple or str, optional
            Spherical grid or path to the file containing the spherical grid,
            by default None.
        angular_resolution : float, optional
            Angular resolution, by default None.
        check_total : bool, optional
            Whether to check the total signal, by default False.
        **interp_kwargs : dict
            Additional interpolation parameters.
        """
        arr = getattr(self, key)
        spherical_grid, N_sph = cartesian_to_spherical_array(
            arr,
            self.grid_xyz,
            spherical_grid=spherical_grid,
            angular_resolution=angular_resolution,
            **interp_kwargs,
        )
        if "xyz" in key:
            sph_key = key.replace("xyz", "sph")
            total_key = key.replace("xyz", "total")
        else:
            sph_key = key + '_sph'
            total_key = key + "_total"
        sph_total_key = f"{sph_key}_total"
        setattr(self, sph_key, N_sph)

        if check_total:
            N_total = integrate_spherical(N_sph, spherical_grid)
            setattr(self, sph_total_key, N_total)

            if not np.isclose(getattr(self, total_key), N_total, rtol=1e-2):
                warn_message = sph_interp_warn.format(
                    total_key, getattr(self, total_key), N_total
                )
                warnings.warn(warn_message)
                _logger.warning(warn_message)

        self.spherical_grid = self.k, self.theta, self.phi = spherical_grid

    def get_photon_spectrum_from_a12(self, a1, a2):
        """
        Calculates the photon spectrum from spectral coefficients a1 and a2.

        Parameters
        ----------
        a1 : numpy.ndarray
            Spectral coefficient a1.
        a2 : numpy.ndarray
            Spectral coefficient a2.

        Returns
        -------
        numpy.ndarray
            Photon spectrum.
        """
        prefactor = 0.5 * epsilon_0 * c / hbar
        k = self.kabs
        N_xyz = ne.evaluate(
            "prefactor * (a1.real**2 + a1.imag**2 + a2.real**2 + a2.imag**2) / k"
        )
        N_xyz[0, 0, 0] = 0.0
        return np.fft.fftshift(N_xyz)
    
    def get_background_xyz(self, add_to_cls_dict=True, bgr_idx=None):
        """
        Calculates the background field spectra on Cartesian grid.

        Parameters
        ----------
        add_to_cls_dict : bool, optional
            Whether to add the background spectra to the class dictionary, 
            by default True.
        bgr_idx : int, optional
            Index of the background field, by default None.

        Returns
        -------
        numpy.ndarray
            Background field spectra.
        """
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
        """
        Calculates the background field spectra.

        Parameters
        ----------
        discernibility : str, optional
            Type of discernibility, by default "angular".
        bgr_idx : int, optional
            Index of the background field, by default None.
        **interp_kwargs : dict
            Additional interpolation parameters.

        Returns
        -------
        numpy.ndarray
            Background field spectra.
        """
        bgr_N_xyz = self.get_background_xyz(add_to_cls_dict=True, bgr_idx=bgr_idx)

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
        """
        Calculates the discernible signal.

        Parameters
        ----------
        discernibility : str, optional
            Type of discernibility, by default "angular".
        """
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
        """
        Calculates the spectral coefficients a1 and a2 from signal amplitudes S1 and S2.

        Parameters
        ----------
        add_signal_bg : bool, optional
            Whether to add the background signal, by default False.
        """
        self.a1_sig, self.a2_sig = transform_S12_to_a12(self.S1, self.S2, self.kabs)
        if add_signal_bg:
            bgr_field = MaxwellMultiple(self.fields_params, self.grid_xyz)
            a1_bgr, a2_bgr = bgr_field.a1, bgr_field.a2
            self.a1_mix, self.a2_mix = self.a1_sig + a1_bgr, self.a2_sig + a2_bgr
        self.a1_sig, self.a2_sig = [np.fft.fftshift(a) for a 
                                    in (self.a1_sig,self.a2_sig)]
        self.a1_mix, self.a2_mix = [np.fft.fftshift(a) for a 
                                    in (self.a1_mix,self.a2_mix)]

    def write_data(self, keys):
        """
        Writes the data to a file.

        Parameters
        ----------
        keys : list of str
            List of keys for the data to be written.
        """
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
        """
        Calculates the total spectra.

        Parameters
        ----------
        calculate_xyz_background : bool, optional
            Whether to calculate the background spectra on Cartesian grid, 
            by default False.
        bgr_idx : int, optional
            Index of the background field, by default None.
        calculate_spherical : bool, optional
            Whether to calculate the spectra on spherical grid,
            by default False.
        spherical_params : dict, optional
            Parameters for the spherical grid, by default None.
        calculate_discernible : bool, optional
            Whether to calculate the discernible signal, by default False.
        discernibility : str, optional
            Type of discernibility, by default "angular".
        """
        self.get_total_signal()
        keys = "kx ky kz N_xyz N_total".split()

        if calculate_spherical:
            self.get_signal_on_sph_grid(key="N_xyz", check_total=True, 
                                        **spherical_params)
            keys.extend("k theta phi N_sph N_sph_total".split())
        if calculate_xyz_background:
            # self.get_background_xyz()
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
        """
        Calculates the polarization spectra.

        Parameters
        ----------
        perp_field_idx : int, optional
            Index of the perpendicular field, by default 1.
        perp_type : str, optional
            Type of perpendicular polarization, by default None.
        calculate_spherical : bool, optional
            Whether to calculate the spectra on spherical grid,
            by default False.
        spherical_params : dict, optional
            Parameters for the spherical grid, by default None.
        stokes : bool, optional
            Whether to calculate Stokes parameters, by default False.
        """
        self.perp_field_idx = perp_field_idx - 1
        angle_keys = "theta phi beta".split()
        angles = [self.fields_params[self.perp_field_idx][key] for key in angle_keys]
        self.get_perp_signal(angles, perp_type=perp_type, stokes=stokes)
        # keys = "kx ky kz Np_xyz Np_total epx epy epz".split()
        keys = "kx ky kz epx epy epz efx efy efz".split()

        if not stokes:
            keys.extend("Np_xyz Np_total".split())
            if calculate_spherical:
                self.get_signal_on_sph_grid(key="Np_xyz", check_total=True,
                                             **spherical_params)
                keys.extend("k theta phi Np_sph Np_sph_total".split())
        else:
            stokes_keys = "P1 P2 P3".split()
            keys.extend(stokes_keys)
            if calculate_spherical:
                keys.extend("k theta phi".split())
                for stokes_key in stokes_keys:
                    self.get_signal_on_sph_grid(key=stokes_key, check_total=False,
                                                **spherical_params)
                    keys.extend([f"{stokes_key}_sph"])
        
        self.write_data(keys)

    def get_mix_signal_bg(self, add_signal_bg=False):
        """
        Calculates the mixed signal and background.

        Parameters
        ----------
        add_signal_bg : bool, optional
            Whether to add the background signal, by default False.
        """
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
        """
        Calculates the spectra based on the specified mode.

        Parameters
        ----------
        mode : str, optional
            Mode of the calculation, by default "total".
        perp_field_idx : int, optional
            Index of the perpendicular field, by default 1.
        perp_type : str, optional
            Type of perpendicular polarization, by default None.
        calculate_xyz_background : bool, optional
            Whether to calculate the background spectra on Cartesian grid,
            by default False.
        bgr_idx : int, optional
            Index of the background field, by default None.
        stokes : bool, optional
            Whether to calculate Stokes parameters, by default False.
        calculate_spherical : bool, optional
            Whether to calculate the spectra on spherical grid,
            by default False.
        spherical_params : dict, optional
            Parameters for the spherical grid, by default None.
        calculate_discernible : bool, optional
            Whether to calculate the discernible signal, by default False.
        discernibility : str, optional
            Type of discernibility, by default "angular".
        add_signal_bg : bool, optional
            Whether to add the background signal, by default False.
        """
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

