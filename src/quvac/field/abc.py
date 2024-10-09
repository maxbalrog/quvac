'''
This script provides abstract interface for existing and future field
classes
'''

from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import numexpr as ne
from scipy.constants import pi, c
import pyfftw

from quvac.field.utils import get_field_energy_kspace


class Field(ABC):
    @abstractmethod
    def calculate_field(self, t, E_out=None, B_out=None, **kwargs):
        '''
        Calculates fields for a given time step
        '''
        ...


class AnalyticField(Field):
    '''
    For such fields analytic formula is known for all time steps,
    every time step the formula is called to calculate the fields
    '''


class MaxwellField(Field):
    '''
    For such fields the initial field distribution (spectral coefficients)
    at a certain time step is given with analytic expression or from file.
    For later time steps the field is propagated according to linear Maxwell 
    equations 
    '''
    def __init__(self):
        self.omega = self.kabs*c
        self.norm_ifft = self.dVk / (2.*pi)**3
        for ax in 'xyz':
            self.__dict__[f'Ef{ax}_expr'] = f"(e1{ax}*a1 + e2{ax}*a2)"
            self.__dict__[f'Bf{ax}_expr'] = f"(e2{ax}*a1 - e1{ax}*a2)"

    def allocate_fft(self):
        self.Ef = [pyfftw.zeros_aligned(self.grid_shape,  dtype='complex128')
                   for _ in range(3)]
        self.Efx, self.Efy, self.Efz = self.Ef
        # pyfftw scheme
        self.Ef_fftw = [pyfftw.FFTW(a, a, axes=(0, 1, 2),
                                    # direction='FFTW_BACKWARD',
                                    direction='FFTW_FORWARD',
                                    flags=('FFTW_MEASURE', ),
                                    threads=1)
                        for a in self.Ef]
    
    def allocate_ifft(self):
        self.EB = [pyfftw.zeros_aligned(self.grid_shape,  dtype='complex128')
                   for _ in range(6)]
        self.prefactor = pyfftw.zeros_aligned(self.grid_shape,  dtype='complex128')
        self.EB_ = [pyfftw.zeros_aligned(self.grid_shape,  dtype='complex128')
                   for _ in range(6)]
        # pyfftw scheme
        self.EB_fftw = [pyfftw.FFTW(a, a, axes=(0, 1, 2),
                                    # direction='FFTW_FORWARD',
                                    direction='FFTW_BACKWARD',
                                    flags=('FFTW_MEASURE', ),
                                    threads=1)
                        for a in self.EB]

    def get_a12(self, E):
        # Calculate Fourier of initial field profile
        # self.allocate_fft()
        # self.Ef = [0,0,0]
        for idx in range(3):
            # self.Ef[idx] *= self.exp_shift_before_fft
            # self.Ef[idx] = pyfftw.interfaces.numpy_fft.fftn(self.Ef[idx], axes=(0,1,2), norm='backward')
            self.Ef_fftw[idx].execute()
            self.Ef[idx] *= self.exp_shift_fft
        # Calculate a1, a2 coefficients
        self.kx, self.ky, self.kz = self.kmeshgrid
        prefactor = self.dV
        # prefactor = ne.evaluate("where((kx==0)&(ky==0)&(kz==0), 0., -1j*dV / kabs)",
        #                         global_dict=self.__dict__)
        self.a1 = ne.evaluate(f"prefactor * (e1x*Efx + e1y*Efy + e1z*Efz)",
                              global_dict=self.__dict__)
        self.a2 = ne.evaluate(f"prefactor * (e2x*Efx + e2y*Efy + e2z*Efz)",
                              global_dict=self.__dict__)

        # Fix energy
        W_upd = get_field_energy_kspace(self.a1, self.a2, self.kabs, self.dVk, mode='without 1/k')
        print(W_upd)
        self.a1 *= np.sqrt(self.W/W_upd) #/ (2*pi)**1.5
        self.a2 *= np.sqrt(self.W/W_upd) #/ (2*pi)**1.5

        # self.a1 *= 1j*self.kabs
        # self.a2 *= 1j*self.kabs

        del self.Ef, E, self.Ef_fftw, self.Efx, self.Efy, self.Efz

        return self.a1, self.a2
    
    def shift_arrays(self):
        to_shift = "a1 a2 kabs omega e1x e1y e1z e2x e2y".split()
        for name in to_shift:
            self.__dict__[name] = np.fft.fftshift(self.__dict__[name])

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
        
        # self.allocate_ifft()
        # Calculate fourier of fields at time t and transform back to 
        # spatial domain
        # prefactor = ne.evaluate("exp(-1j*omega*t) * 1j*kabs", global_dict=self.__dict__)
        ne.evaluate("exp(-1j*omega*(t-t0))", global_dict=self.__dict__,
                    out=self.prefactor)
        # ne.evaluate("exp(-1j*omega*t) * 1j*kabs", global_dict=self.__dict__,
        #             out=self.prefactor)
        for idx in range(6):
            field_comp = self.EB_[idx] #* self.exp_shift_before_ifft
            # field_comp = self.__dict__[f'{field}f{ax}_expr']
            ne.evaluate(f"prefactor * field_comp", global_dict=self.__dict__,
                        out=self.EB[idx])
            self.EB_fftw[idx].execute()
            # self.EB[idx] *= self.norm_ifft
            self.EB[idx] = self.EB[idx] * self.norm_ifft * self.exp_shift_ifft
        # for i,field in enumerate('EB'):
        #     for j,ax in enumerate('xyz'):
        #         idx = 3*i + j
        #         field_comp = self.EB_[idx]
        #         # field_comp = self.__dict__[f'{field}f{ax}_expr']
        #         ne.evaluate(f"prefactor * field_comp", global_dict=self.__dict__,
        #                     out=self.EB[idx])
        #         self.EB_fftw[idx].execute()
        
        # problem that one should add fields in complex domain?
        # for idx,(Ei,Bi) in enumerate(zip(E_out, B_out)):
        for idx in range(3):
            E_out[idx] += np.fft.fftshift(self.EB[idx])
            B_out[idx] += np.fft.fftshift(self.EB[3+idx])
            # Ei += np.real(self.EB[idx])*self.norm_ifft
            # Bi += np.real(self.EB[3+idx])*self.norm_ifft
            # Ei += np.real(self.EB[idx])
            # Bi += np.real(self.EB[3+idx])
        return E_out, B_out


class FieldFromFile(Field):
    '''
    (???) Potentially for fields that are pre-calculated somewhere else and
    are loaded from file
    '''


