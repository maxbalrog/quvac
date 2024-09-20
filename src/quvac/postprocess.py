'''
Here we provide analyzer classes that calculate from amplitudes:
    - Total (polarization insensitive) signal
    - Polarization sensitive signal
    - Discernible signal
'''

import numpy as np
import numexpr as ne
from scipy.constants import pi

from quvac.grid import GridXYZ


class VacuumEmissionAnalyzer:
    '''
    
    '''
    def __init__(self, data_path, save_path=None):
        # Load data
        self.data = np.load(data_path)
        grid = tuple(self.data['x'], self.data['y'], self.data['z'])
        self.grid = GridXYZ(grid)
        self.grid.get_k_grid()
        for ax in 'xyz':
            self.__dict__[f'k{ax}'] = np.fft.fftshift(self.__dict__[f'k{ax}'])
        # Update local dict with variables from GridXYZ class
        self.__dict__.update(self.grid.__dict__)

        self.S1, self.S2 = self.data['S1'], self.data['S2']

        self.save_path = save_path

    def get_total_signal_spectrum(self):
        self.S = ne.evaluate("S1.real**2 + S1.imag**2 + S2.real**2 + S2.imag**2",
                             global_dict=self.__dict__)
        self.N_xyz = np.fft.fftshift(self.S / (2*pi)**3)

    def get_total_signal(self):
        Ntot = ne.evaluate("sum(N_xyz * dVk)",
                           global_dict=self.__dict__)
        return Ntot
    
    def get_pol_signal(self, pol_vector):
        px, py, pz = pol_vector

    def get_signal_spherical(self):
        pass

    def get_discernible_signal(self):
        pass

