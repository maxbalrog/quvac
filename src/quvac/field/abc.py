"""
This script provides abstract interface for existing and future field
classes
"""

import logging
from abc import ABC, abstractmethod

import numexpr as ne
import numpy as np
import pyfftw

from quvac.field.utils import get_field_energy_kspace

logger = logging.getLogger("simulation")


class Field(ABC):
    @abstractmethod
    def calculate_field(self, t, E_out=None, B_out=None, **kwargs):
        """
        Calculates fields for a given time step
        """
        ...


class ExplicitField(Field):
    """
    For such fields analytic formula is known for all time steps.
    One time step (e.g., focus at t=0) can serve as model field
    for the calculation of Maxwell coefficients and Maxwell propagation
    to other time steps

    Parameters:
    -----------
    grid: quvac.grid.GridXYZ
        spatial and spectral grid
    """

    def __init__(self, grid):
        self.grid_xyz = grid
        self.__dict__.update(self.grid_xyz.__dict__)

        # Define FFT shift
        k_grid = [np.fft.fftshift(kx) for kx in self.kgrid]
        kmeshgrid_shift = np.meshgrid(*k_grid, indexing="ij", sparse=True)
        exp_shift_after_fft = sum(
            [(kx - kx.flatten()[0]) * x[0] for kx, x in zip(kmeshgrid_shift, self.grid)]
        )
        self.exp_shift_after_fft = ne.evaluate("exp(-1j*exp_shift_after_fft)")

    def allocate_fft(self):
        self.Ef = [
            pyfftw.zeros_aligned(self.grid_shape, dtype="complex128") for _ in range(3)
        ]
        # pyfftw scheme
        self.Ef_fftw = [
            pyfftw.FFTW(
                a,
                a,
                axes=(0, 1, 2),
                direction="FFTW_FORWARD",
                flags=("FFTW_MEASURE",),
                threads=1,
            )
            for a in self.Ef
        ]

    def get_a12(self, t0=None):
        t0 = t0 if t0 is not None else self.t0
        self.allocate_fft()
        self.calculate_field(t0, E_out=self.Ef, mode="complex")

        for idx in range(3):
            self.Ef_fftw[idx].execute()
            self.Ef[idx] *= self.exp_shift_after_fft

        # Calculate a1, a2 coefficients
        Efx, Efy, Efz = self.Ef

        self.a1 = ne.evaluate(
            f"dV * (e1x*Efx + e1y*Efy + e1z*Efz)", global_dict=self.__dict__
        )
        self.a2 = ne.evaluate(
            f"dV * (e2x*Efx + e2y*Efy + e2z*Efz)", global_dict=self.__dict__
        )

        # Fix energy
        W_upd = get_field_energy_kspace(
            self.a1, self.a2, self.kabs, self.dVk, mode="without 1/k"
        )
        logger.info(f"    Energy after projection in k-space: {W_upd:.3f} J")

        self.a1 *= np.sqrt(self.W / W_upd)
        self.a2 *= np.sqrt(self.W / W_upd)

        W_corrected = get_field_energy_kspace(
            self.a1, self.a2, self.kabs, self.dVk, mode="without 1/k"
        )
        logger.info(f'    Energy after "correction":          {W_corrected:.3f} J')

        del self.Ef, self.Ef_fftw
        return self.a1, self.a2


class FieldFromFile(Field):
    """
    (???) Potentially for fields that are pre-calculated somewhere else and
    are loaded from file
    """
