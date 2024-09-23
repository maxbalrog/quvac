'''
Here we provide analyzer classes that calculate from amplitudes:
    - Total (polarization insensitive) signal
    - Polarization sensitive signal
    - Discernible signal
'''

import numpy as np
import numexpr as ne
from scipy.constants import pi
from scipy.interpolate import RegularGridInterpolator
from astropy.coordinates import cartesian_to_spherical

from quvac.grid import GridXYZ
from quvac.grid_utils import get_pol_basis


def get_polarization_vector(theta, phi, beta):
    e1, e2 = get_pol_basis(theta, phi)
    ep = e1*np.cos(beta) + e2*np.sin(beta)
    return ep


def cartesian_to_spherical_ax(x, y, z):
    '''
    Transforms the cartesian grid to spherical grid
    '''
    if x.ndim == 1:
        x = x.reshape((-1,1,1))
        y = y.reshape((1,-1,1))
        z = z.reshape((1,1,-1))
    sph = cartesian_to_spherical(x, y, z)
    r,theta,phi = [np.array(ax) for ax in sph]
    theta += pi/2
    return r, theta, phi


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
    spherical_mesh = np.meshgrid(*spherical_grid)

    # Find corresponding spherical coordinates of cartesian grid
    sph_ax = cartesian_to_spherical_ax(*xyz_grid.grid)

    # Build interpolator
    arr_interp = RegularGridInterpolator(sph_ax, arr, fill_value=0.,
                                         **interp_kwargs)

    # Interpolate data on a desired grid
    arr_sph = arr_interp(*spherical_mesh)
    return spherical_grid, arr_sph


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
    def __init__(self, data_path, save_path=None):
        # Load data
        self.data = np.load(data_path)
        grid = tuple((self.data['x'], self.data['y'], self.data['z']))
        self.grid_ = GridXYZ(grid)
        self.grid_.get_k_grid()
        # Update local dict with variables from GridXYZ class
        self.__dict__.update(self.grid_.__dict__)

        for ax in 'xyz':
            self.__dict__[f'k{ax}'] = np.fft.fftshift(self.__dict__[f'k{ax}'])

        self.S1, self.S2 = self.data['S1'], self.data['S2']

        self.save_path = save_path

    def get_total_signal_spectrum(self):
        self.S = ne.evaluate("S1.real**2 + S1.imag**2 + S2.real**2 + S2.imag**2",
                             global_dict=self.__dict__)
        self.N_xyz = np.fft.fftshift(self.S / (2*pi)**3)

    def get_total_signal(self):
        self.N_tot = ne.evaluate("sum(N_xyz * dVk)",
                           global_dict=self.__dict__)
    
    def get_pol_signal_spectrum(self, angles):
        '''
        angles (theta, phi, beta): (float, float, float)
            Euler angles for field polarization (in degrees)
        '''
        angles = [angle*pi/180 for angle in angles]
        self.ep = epx, epy, epz = get_polarization_vector(*angles)
        ep_e1 = "(epx*e1x + epy*e1y + epz*e1z)"
        ep_e2 = "(epx*e2x + epy*e2y)"
        self.Sp = ne.evaluate(f"abs({ep_e1}*S1 + {ep_e2}*S2)**2", global_dict=self.__dict__)
        self.Np_xyz = np.fft.fftshift(self.Sp / (2*pi)**3)

    def get_pol_signal(self):
        self.Np_tot = ne.evaluate("sum(Np_xyz * dVk)",
                           global_dict=self.__dict__)

    def get_signal_spherical(self, spherical_grid=None, angular_resolution=None, 
                             **interp_kwargs):
        spherical_grid, N_sph = cartesian_to_spherical_array(self.N_xyz, self.grid_,
                                                             spherical_grid=spherical_grid,
                                                             angular_resolution=angular_resolution, 
                                                            **interp_kwargs)
        self.k, self.theta, self.phi = spherical_grid
        self.N_sph = N_sph
        if self.__dict__.get('Np_xyz', None) is not None:
            _, self.Np_sph = cartesian_to_spherical_array(self.Np_xyz, self.grid_,
                                                          spherical_grid=spherical_grid,
                                                          angular_resolution=None, 
                                                          **interp_kwargs)
        else:
            self.Np_sph = None
        return self.N_sph, self.Np_sph
    
    def write_data(self):
        data = {
            'kx': self.kx,
            'ky': self.ky,
            'kz': self.kz,
            'N_xyz': self.N_xyz,
            'N_tot': self.N_tot,
            'ep': self.ep,
            'Np_xyz': self.Np_xyz,
            'Np_tot': self.Np_tot
        }
        if self.__dict__.get('N_sph', None) is not None:
            data.update({
                'k': self.k,
                'theta': self.theta,
                'phi': self.phi,
                'N_sph': self.N_sph
            })
        np.savez(self.save_path, **data)
    
    def get_spectra(self, angles):
        self.get_total_signal_spectrum()
        self.get_total_signal()

        self.get_pol_signal_spectrum(angles)
        self.get_pol_signal()

        # self.get_signal_spherical()
        
        self.write_data()
        

    def get_discernible_signal(self):
        pass

