'''
This script implements Grid class calculating necessary variables
for spatial grid and its Fourier counterpart
'''

import numpy as np
import numexpr as ne
from scipy.constants import pi


class GridXYZ(object):
    '''
    Calculates space grid and Fourier counterpart
    '''
    def __init__(self, grid):
        # Define spatial grid
        self.grid = grid
        self.grid_shape = tuple(ax.size for ax in grid)
        self.N = np.prod(self.grid_shape)
        self.xyz = self.x, self.y, self.z = np.meshgrid(*grid, indexing='ij', sparse=True)
        self.dxyz = tuple(ax[1]-ax[0] for ax in grid)
        self.dV = np.prod(self.dxyz)

    def get_k_grid(self):
        for i,ax in enumerate('xyz'):
            self.__dict__[f'k{ax}'] = 2.*pi*np.fft.fftfreq(self.grid_shape[i],
                                                               self.dxyz[i])
        self.kgrid = tuple((self.kx, self.ky, self.kz))
        self.dkxkykz = [ax[1]-ax[0] for ax in self.kgrid]
        self.dVk = np.prod(self.dkxkykz)

        self.kmeshgrid = np.meshgrid(self.kx, self.ky, self.kz, indexing='ij', sparse=True)
        kx, ky, kz = self.kmeshgrid
        self.kabs = kabs = ne.evaluate("sqrt(kx**2 + ky**2 + kz**2)")
        kperp = ne.evaluate("sqrt(kx**2 + ky**2)")

        # Polarization vectors
        self.e1x = ne.evaluate("where((kx==0) & (ky==0), 1.0, kx * kz / (kperp*kabs))")
        self.e1y = ne.evaluate("where((kx==0) & (ky==0), 0.0, ky * kz / (kperp*kabs))")
        self.e1z = ne.evaluate("where((kx==0) & (ky==0), 0.0, -kperp / kabs)")

        self.e2x = ne.evaluate("where((kx==0) & (ky==0), 0.0, -ky / kperp)")
        self.e2y = ne.evaluate("where((kx==0) & (ky==0), 1.0, kx / kperp)")
        self.e2z = 0.