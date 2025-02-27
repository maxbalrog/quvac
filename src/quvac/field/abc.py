"""
Abstract interfaces for field classes.
"""

import logging
from abc import ABC, abstractmethod

import numexpr as ne
import numpy as np
import pyfftw

from quvac.field.utils import get_field_energy_kspace

_logger = logging.getLogger("simulation")


class Field(ABC):
    """
    Abstract base class for fields.

    This class defines the interface for calculating fields at a given time step.
    Subclasses must implement the ``calculate_field`` method.

    Methods
    -------
    calculate_field(t, E_out=None, B_out=None, **kwargs)
        Calculates fields for a given time step.
    """
    @abstractmethod
    def calculate_field(self, t, E_out=None, B_out=None, **kwargs):
        """
        Calculates fields for a given time step.

        Parameters
        ----------
        t : float
            The time step at which to calculate the fields.
        E_out : array-like, optional
            Output array for the electric field.
        B_out : array-like, optional
            Output array for the magnetic field.
        **kwargs : dict
            Additional parameters for field calculation.

        Returns
        -------
        E_out, B_out : array-like
            The electric and magnetic fields at the given time step.
        """
        ...


class ExplicitField(Field):
    """
    For fields with an analytic formula known in the whole space at 
    least in the focus (t=0).

    One time step (e.g., focus at t=0) can serve as a model field
    for the calculation of Maxwell coefficients and Maxwell propagation
    to other time steps.

    Parameters
    ----------
    grid : quvac.grid.GridXYZ
        Spatial and spectral grid.

    Attributes
    ----------
    grid_xyz : quvac.grid.GridXYZ
        The spatial and spectral grid.
    Ef : list of complex numpy.ndarray
        The electric field in Fourier space.
    Ef_fftw : list of pyfftw.FFTW
        The FFTW objects for the electric field.
    a1 : complex numpy.ndarray
        The a1 spectral coefficient.
    a2 : complex numpy.ndarray
        The a2 spectral coefficient.

    Methods
    -------
    allocate_fft()
        Allocates memory for FFT calculations.
    get_a12(t0=None)
        Calculates the a1 and a2 coefficients at a given time step.
    """

    def __init__(self, grid):
        self.grid_xyz = grid
        self.__dict__.update(self.grid_xyz.__dict__)

    def allocate_fft(self):
        """
        Allocates memory for FFT calculations.
        """
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
        """
        Calculates the a1 and a2 coefficients at a given time step.

        Parameters
        ----------
        t0 : float, optional
            The time step at which to calculate the coefficients. If None, uses self.t0.

        Returns
        -------
        a1 : numpy.ndarray
            The a1 spectral coefficient.
        a2 : numpy.ndarray
            The a2 spectral coefficient.

        Notes
        -----
        After the projection of the field on spectral coefficients, its energy
        is corrected to the desired value.
        """
        t0 = t0 if t0 is not None else self.t0
        self.allocate_fft()
        self.calculate_field(t0, E_out=self.Ef, mode="complex")

        for idx in range(3):
            self.Ef_fftw[idx].execute()

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
        _logger.info(f"    Energy after projection in k-space: {W_upd:.3f} J")

        self.a1 *= np.sqrt(self.W / W_upd)
        self.a2 *= np.sqrt(self.W / W_upd)

        W_corrected = get_field_energy_kspace(
            self.a1, self.a2, self.kabs, self.dVk, mode="without 1/k"
        )
        _logger.info(f'    Energy after "correction":          {W_corrected:.3f} J')

        del self.Ef, self.Ef_fftw
        return self.a1, self.a2


class FieldFromFile(Field):
    """
    For fields that are loaded from file.

    Not implemented yet.
    """
