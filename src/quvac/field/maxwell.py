'''
This script provides basic linear Maxwell propagation class
and a particular implementation of ParaxialGaussianMaxwell
'''
import os

import numpy as np
import numexpr as ne
from scipy.constants import pi, c
import pyfftw

from quvac.field.abc import MaxwellField
from quvac.field.paraxial_gaussian import ParaxialGaussianAnalytic


class ParaxialGaussianMaxwell(MaxwellField):
    '''
    Initial field at focus as paraxial gaussian and propagate
    to later timesteps according to linear Maxwell equations
    '''
    def __init__(self, field_params, grid, nthreads=None):
        self.grid = grid
        self.grid.get_k_grid()
        self.__dict__.update(self.grid.__dict__)

        k_grid = [np.fft.fftshift(kx) for kx in self.kgrid]
        phi0 = sum([x[0]*kx[0] for x,kx in zip(self.grid, k_grid)])

        # self.exp_shift = sum([kx[0]*x for kx,x in zip(k_grid, self.xyz)])
        # self.exp_shift = ne.evaluate('exp(-1j*exp_shift)', global_dict=self.__dict__)

        self.exp_shift_fft = sum([np.fft.fftshift(kx)*x.flatten()[0] for kx,x in zip(self.kmeshgrid, self.xyz)])
        self.exp_shift_fft = ne.evaluate('exp(-1j*exp_shift_fft)', 
                                           global_dict=self.__dict__)
        
        self.exp_shift_ifft = sum([kx[0]*np.fft.fftshift(x) for kx,x in zip(k_grid, self.xyz)])
        self.exp_shift_ifft = ne.evaluate('exp(1j*exp_shift_ifft)', 
                                           global_dict=self.__dict__)
        
        self.exp_shift_before_fft = ne.evaluate('exp(-1j*exp_shift_ifft)', 
                                           global_dict=self.__dict__)
        
        self.exp_shift_before_ifft = ne.evaluate('exp(1j*exp_shift_fft)', 
                                           global_dict=self.__dict__)

        self.nthreads = nthreads if nthreads else os.cpu_count()

        # Initialize base class
        super().__init__()
        self.allocate_fft()

        self.W = 0.

        # Initialize the analytic class and calculate ini field
        self.E_ini = [pyfftw.zeros_aligned(self.grid_shape,  dtype='complex128')
                      for _ in range(3)]
        if isinstance(field_params, dict):
            print('Solo parameter')
            field_params = [field_params]
        for param in field_params:
            ini_field = ParaxialGaussianAnalytic(param, grid)
            self.t0 = ini_field.t0
            self.W += ini_field.W
            ini_field.calculate_field(self.t0, E_out=self.Ef, mode='complex')

        for i in range(3):
            self.E_ini[i] += self.Ef[i].copy()
        # Get a1,a2 coefficients
        self.get_a12(self.E_ini)
        # self.shift_arrays()
        self.allocate_ifft()
        self.get_fourier_fields()

    def calculate_field(self, t, E_out=None, B_out=None):
        return super().calculate_field(t, E_out, B_out)


class ParaxialGaussianMaxwellMultiple(MaxwellField):

    def __init__(self, field_params, grid, nthreads=None):
        self.grid = grid
        self.grid.get_k_grid()
        self.__dict__.update(self.grid.__dict__)

        self.nthreads = nthreads if nthreads else os.cpu_count()
        print('We enter multiple gaussian')

        # Initialize base class
        super().__init__()
        self.allocate_fft()

        self.a1, self.a2 = [pyfftw.zeros_aligned(self.grid_shape,  dtype='complex128')
                            for _ in range(2)]
        self.E_ini = [pyfftw.zeros_aligned(self.grid_shape,  dtype='complex128')
                      for _ in range(3)]

        for param in field_params:
            print('Ho-ho')
            field = ParaxialGaussianMaxwell(param, grid, nthreads)
            self.t0 = field.t0
            self.a1 += field.a1
            self.a2 += field.a2
            for i in range(3):
                self.E_ini[i] += field.E_ini[i]
        
        keys = 'exp_shift_before_ifft exp_shift_ifft'.split()
        self.__dict__.update({k:v for k,v in field.__dict__.items() 
                              if k in keys})
        # self.__dict__.update(field.__dict__)
        
        self.allocate_ifft()
        self.get_fourier_fields()

    def calculate_field(self, t, E_out=None, B_out=None):
        return super().calculate_field(t, E_out, B_out)


        

