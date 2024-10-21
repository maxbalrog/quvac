'''
Here we provide analyzer classes that calculate from amplitudes:
    - Total (polarization insensitive) signal
    - Polarization sensitive signal
    - Discernible signal
'''
import warnings

import numpy as np
import numexpr as ne
from scipy.constants import pi, c, hbar, epsilon_0
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid

from quvac.grid import GridXYZ, get_pol_basis
from quvac.field.maxwell import MaxwellMultiple


def get_polarization_vector(theta, phi, beta):
    e1, e2 = get_pol_basis(theta, phi)
    ep = e1*np.cos(beta) + e2*np.sin(beta)
    return ep


def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cartesian_to_spherical_array(arr, xyz_grid, spherical_grid=None,
                                 angular_resolution=None, 
                                 **interp_kwargs):
    '''
    Transforms an array with data on cartesian grid to the
    array with data on spherical grid
    '''
    # Calculate spherical grid if not given
    if not spherical_grid:
        dk = np.min(xyz_grid.dkxkykz)
        kmax = np.max(xyz_grid.kabs)
        dangle = angular_resolution if angular_resolution else 1.*pi/180

        k = np.arange(0., kmax, dk)
        theta = np.arange(0., pi, dangle)
        phi = np.arange(0., 2*pi, dangle)
        spherical_grid = (k, theta, phi)
    spherical_mesh = np.meshgrid(*spherical_grid, indexing='ij')
    nk, ntheta, nphi = [len(ax) for ax in spherical_grid]

    # Find corresponding cartesian coordinates of spherical mesh:
    # (r,theta,phi) -> (x, y, z)
    xyz_for_sph = sph2cart(*spherical_mesh)
    xyz_for_sph_pts = np.vstack([ax.flatten() for ax in xyz_for_sph]).T

    # Build interpolator (x,y,z) -> arr
    arr_interp = RegularGridInterpolator(xyz_grid.kgrid_shifted, arr, fill_value=0.,
                                         bounds_error=False, **interp_kwargs)

    # Interpolate data on a desired grid
    arr_sph = arr_interp(xyz_for_sph_pts).reshape((nk, ntheta, nphi))
    return spherical_grid, arr_sph


def integrate_spherical(arr, axs, axs_names=['k','theta','phi'],
                        axs_integrate=['k','theta','phi']):
    err_msg = 'Axes of array and axs names do not match'
    assert len(axs) == len(axs_names) and len(axs) >= len(axs_integrate), err_msg

    axs_names_ = axs_names.copy()

    integrand = arr.copy()
    for ax_name in axs_integrate:
        idx = axs_names.index(ax_name)
        idx_ = axs_names_.index(ax_name)

        ax_shape = [1 for _ in range(len(axs_names_))]
        ax_shape[idx_] = integrand.shape[idx_]

        ax = axs[idx].reshape(ax_shape)
        if ax_name == 'k':
            integrand *= ax**2
        elif ax_name == 'theta':
            integrand *= np.sin(ax)
        integrand = trapezoid(integrand, axs[idx], axis=idx_)
        axs_names_.pop(idx_)
    return integrand


class VacuumEmissionAnalyzer:
    '''
    Calculates spectra and observables from amplitudes
    provided by quvac.integrator.vacuum_emission.VacuumEmission
    class

    Currently supports:
        - Differential polarization-(in)sensitive spectrum on (kx,ky,kz) grid
        - Differential polarization-(in)sensitive spectrum on (k,theta,phi) grid
        - Total signal
    '''
    def __init__(self, fields_params, data_path, save_path=None):
        self.fields_params = fields_params
        # Load data
        self.data = np.load(data_path)
        grid = tuple((self.data['x'], self.data['y'], self.data['z']))
        self.grid_xyz = GridXYZ(grid)
        self.grid_xyz.get_k_grid()
        # Update local dict with variables from GridXYZ class
        self.__dict__.update(self.grid_xyz.__dict__)

        for ax in 'xyz':
            self.__dict__[f'k{ax}'] = np.fft.fftshift(self.__dict__[f'k{ax}'])

        self.S1, self.S2 = self.data['S1'], self.data['S2']

        self.save_path = save_path

    def get_total_signal_spectrum(self):
        self.S = ne.evaluate("S1.real**2 + S1.imag**2 + S2.real**2 + S2.imag**2",
                             global_dict=self.__dict__)
        self.N_xyz = np.fft.fftshift(self.S / (2*pi)**3)

        self.N_tot = ne.evaluate("sum(N_xyz)",
                           global_dict=self.__dict__)
        self.N_tot *= self.dVk
    
    def get_pol_signal_spectrum(self, angles):
        '''
        angles (theta, phi, beta): (float, float, float)
            Euler angles for field polarization (in degrees)
        '''
        angles = [angle*pi/180 for angle in angles]
        self.ep = epx, epy, epz = get_polarization_vector(*angles)
        ep_e1 = "(epx*e1x + epy*e1y + epz*e1z)"
        ep_e2 = "(epx*e2x + epy*e2y + epz*e2z)"
        self.Sp = ne.evaluate(f"({ep_e1}*S1 + {ep_e2}*S2)", global_dict=self.__dict__)
        self.Sp = ne.evaluate('Sp.real**2 + Sp.imag**2', global_dict=self.__dict__)
        self.Np_xyz = np.fft.fftshift(self.Sp / (2*pi)**3)

        self.Np_tot = ne.evaluate("sum(Np_xyz)",
                                  global_dict=self.__dict__)
        self.Np_tot *= self.dVk

    def get_signal_on_sph_grid(self, spherical_grid=None, angular_resolution=None, 
                               **interp_kwargs):
        for key in 'N_xyz Np_xyz'.split():
            if key in self.__dict__:
                arr = self.__dict__[key]
                spherical_grid, N_sph = cartesian_to_spherical_array(arr, self.grid_xyz,
                                                                    spherical_grid=spherical_grid,
                                                                    angular_resolution=angular_resolution, 
                                                                    **interp_kwargs)
                sph_key = key.replace('xyz', 'sph')
                sph_total_key = f'{sph_key}_tot'
                total_key = key.replace('xyz', 'tot')
                self.__dict__[sph_key] = N_sph

                N_total = integrate_spherical(N_sph, spherical_grid)
                self.__dict__[sph_total_key] = N_total

                if not np.isclose(self.__dict__[total_key], N_total, rtol=1e-2):
                    warnings.warn(f"""{total_key} signal on cartesian and spherical 
                                  grid differ by more than 1%:
                                  N total (xyz): {self.__dict__[total_key]:.3f}
                                  N total (sph): {N_total:.3f}""")

        self.spherical_grid = self.k, self.theta, self.phi = spherical_grid

    def get_photon_spectrum_from_a12(self, a1, a2):
        prefactor = 0.5 * epsilon_0 * c / hbar
        k = self.kabs
        N_xyz = ne.evaluate('prefactor * (a1.real**2 + a1.imag**2 + a2.real**2 + a2.imag**2) / k')
        N_xyz[0, 0, 0] = 0.
        return np.fft.fftshift(N_xyz)
    
    def get_background(self, discernibility='angular', **interp_kwargs):
        bgr_field = MaxwellMultiple(self.fields_params, self.grid_xyz)
        bgr_N_xyz = self.get_photon_spectrum_from_a12(bgr_field.a1, bgr_field.a2)

        # Interpolate on spherical grid
        _, bgr_N_sph = cartesian_to_spherical_array(bgr_N_xyz, self.grid_xyz,
                                                    spherical_grid=self.spherical_grid,
                                                    **interp_kwargs)
        
        # Integrate over k if needed
        if discernibility == 'angular':
            bgr_N_sph = integrate_spherical(bgr_N_sph, self.spherical_grid, axs_integrate=['k'])
        return bgr_N_sph

    def get_discernible_signal(self, discernibility='angular'):
        # Calculate numerical background
        self.N_bgr = self.get_background(discernibility=discernibility)

        # Integrate signal spectrum if required and determine discernible regions
        if discernibility == 'angular':
            self.N_angular = integrate_spherical(self.N_sph, self.spherical_grid, axs_integrate=['k'])
            self.discernible = self.N_angular > self.N_bgr
        else:
            self.discernible = self.N_sph > self.N_bgr

        # Integrate over discernible regions
        self.N_disc = integrate_spherical(self.N_sph * self.discernible, self.spherical_grid)

    
    def write_data(self):
        data = {
            'kx': self.kx,
            'ky': self.ky,
            'kz': self.kz,
            'N_xyz': self.N_xyz,
            'N_total': self.N_tot,
            'ep': self.ep,
            'Np_xyz': self.Np_xyz,
            'Np_total': self.Np_tot
        }
        if 'N_sph' in self.__dict__:
            data.update({
                'k': self.k,
                'theta': self.theta,
                'phi': self.phi,
                'N_sph': self.N_sph,
                'N_sph_total': self.N_sph_tot,
            })
        if 'N_disc' in self.__dict__:
            data.update({
                'background': self.N_bgr,
                'discernible': self.discernible,
                'N_disc': self.N_disc,
            })
        if 'N_angular' in self.__dict__:
            data.update({
                'N_angular': self.N_angular,
            })
        np.savez(self.save_path, **data)
    
    def get_spectra(self, angles=None, calculate_spherical=False,
                    calculate_discernible=False):
        self.get_total_signal_spectrum()

        angles = (0.,0.,0.) if angles is None else angles
        self.get_pol_signal_spectrum(angles)

        if calculate_spherical:
            self.get_signal_on_sph_grid()
        if calculate_discernible:
            self.get_discernible_signal()
        
        self.write_data()

