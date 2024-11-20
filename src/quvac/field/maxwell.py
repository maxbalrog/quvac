'''
This script provides basic linear Maxwell propagation class
and a particular implementation of GaussianMaxwell
'''
import logging

import numpy as np
import numexpr as ne
from scipy.constants import pi, c
import pyfftw

from quvac.field.abc import Field
from quvac.field.gaussian import GaussianAnalytic


SPATIAL_MODEL_FIELDS = {
    'paraxial_gaussian_maxwell': GaussianAnalytic,
}


logger = logging.getLogger('simulation')


class MaxwellField(Field):
    '''
    For such fields the initial field distribution (spectral coefficients)
    at a certain time step is given with analytic expression or from file.
    For later time steps the field is propagated according to linear Maxwell 
    equations

    Parameters:
    -----------
    grid: quvac.grid.GridXYZ
        spatial and spectral grid
    '''
    def __init__(self, grid):
        self.grid_xyz = grid
        self.__dict__.update(self.grid_xyz.__dict__)

        self.omega = self.kabs*c
        self.norm_ifft = self.dVk / (2.*pi)**3
        for ax in 'xyz':
            self.__dict__[f'Ef{ax}_expr'] = f"(e1{ax}*a1 + e2{ax}*a2)"
            self.__dict__[f'Bf{ax}_expr'] = f"(e2{ax}*a1 - e1{ax}*a2)"

    def allocate_ifft(self):
        self.EB = [pyfftw.zeros_aligned(self.grid_shape, dtype='complex128')
                   for _ in range(6)]
        self.EB_ = [pyfftw.zeros_aligned(self.grid_shape, dtype='complex128')
                   for _ in range(6)]
        # pyfftw scheme
        self.EB_fftw = [pyfftw.FFTW(a, a, axes=(0, 1, 2),
                                    direction='FFTW_BACKWARD',
                                    flags=('FFTW_MEASURE', ),
                                    threads=1)
                        for a in self.EB]

    def get_fourier_fields(self):
        for i,field in enumerate('EB'):
            for j,ax in enumerate('xyz'):
                idx = 3*i + j
                ne.evaluate(self.__dict__[f'{field}f{ax}_expr'], global_dict=self.__dict__,
                            out=self.EB_[idx])

    def calculate_field(self, t, E_out=None, B_out=None):
        if E_out is None:
            E_out = [np.zeros(self.grid_shape, dtype=np.complex128) for _ in range(3)]
            B_out = [np.zeros(self.grid_shape, dtype=np.complex128) for _ in range(3)]
        
        # Calculate fourier of fields at time t and transform back to 
        # spatial domain
        # ========================================================================
        prefactor = ne.evaluate("exp(-1.j*omega*(t-t0))", global_dict=self.__dict__)
        for idx in range(6):
            ne.evaluate(f"prefactor * EB", global_dict={'EB': self.EB_[idx]},
                        out=self.EB[idx])
            self.EB_fftw[idx].execute()
        # ========================================================================

        
        for idx in range(3):
            E_out[idx] += self.EB[idx] * self.norm_ifft
            B_out[idx] += self.EB[3+idx] * self.norm_ifft
        return E_out, B_out


class MaxwellMultiple(MaxwellField):
    '''
    Calculate spectral coefficients from several fields and
    propagate them
    '''
    def __init__(self, fields, grid, nthreads=None):
        super().__init__(grid)

        self.a1, self.a2 = [pyfftw.zeros_aligned(self.grid_shape,  dtype='complex128')
                            for _ in range(2)]
        self.fields = [fields] if isinstance(fields, dict) else fields
        for i,field in enumerate(self.fields):
            logger.info(f'Setting up field {i+1}:')
            a1, a2 = self.get_a12_from_field(field)
            self.a1 += a1
            self.a2 += a2

        self.allocate_ifft()
        self.get_fourier_fields()

    def get_a12_from_field(self, field_params):
        field_type = field_params['field_type']
        if field_type in SPATIAL_MODEL_FIELDS:
            cls = SPATIAL_MODEL_FIELDS[field_type]
            logger.info(f'    {field_type}: {cls.__name__}')
            ini_field = cls(field_params, self.grid_xyz)
            self.t0 = ini_field.t0
            a1, a2 = ini_field.get_a12(ini_field.t0)
        return a1, a2

    def calculate_field(self, t, E_out=None, B_out=None):
        return super().calculate_field(t, E_out, B_out)

