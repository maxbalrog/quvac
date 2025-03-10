"""
Abstract interfaces for field classes.

For details about spectral coefficient calculation (a1, a2)
refer to :ref:`implementation` section of documentation. 
"""

import logging
from abc import ABC, abstractmethod

import numexpr as ne
import numpy as np
from scipy.constants import pi
from scipy.spatial.transform import Rotation
import pyfftw

from quvac.field.utils import get_field_energy_kspace
from quvac import config

ANGLE_KEYS = "theta phi beta phase0".split()
_logger = logging.getLogger("simulation")


class Field(ABC):
    """
    Abstract base class for fields.

    This class defines the interface for calculating fields at a given time step.
    Subclasses must implement the ``calculate_field`` method.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters.
    grid : quvac.grid.GridXYZ
        Spatial and spectral grid.

    Attributes
    ----------
    grid_xyz : quvac.grid.GridXYZ
        The spatial and spectral grid.
    rotation_m : np.array
        Rotation matrix.
    rotation_bwd_m : np.array
        Inverse rotation matrix.
    """
    def __init__(self, field_params, grid):
        # add grid attributes to field class attributes
        self.grid_xyz = grid
        self.__dict__.update(self.grid_xyz.__dict__)

        # Dynamically create class instance variables available with
        # self.<variable_name>
        for key, val in field_params.items():
            if key in ANGLE_KEYS:
                val *= pi / 180.0
            setattr(self, key, val)
    
    def get_rotation(self):
        """
        Defines the rotation transforming (0,0,1) -> (kx,ky,kz) 
        for vectors and (1,0,0) -> e(beta) = e1*cos(beta) + e2*sin(beta).
        """
        self.beta = getattr(self, "beta", 0)
        self.rotation = Rotation.from_euler("ZYZ", (self.phi, self.theta, self.beta))
        self.rotation_m = self.rotation.as_matrix()
        # Inverse rotation: (kx,ky,kz) -> (0,0,1)
        self.rotation_bwd = self.rotation.inv()
        self.rotation_bwd_m = self.rotation_bwd.as_matrix()

    def rotate_coordinates(self, rotate_grid=True):
        """
        Rotates the coordinate grid.

        Parameters
        ----------
        rotate_grid : bool, optional
            Whether to rotate the grid coordinates. Default is True.
        """
        self.get_rotation()
        if rotate_grid:
            axes = "xyz"
            x_, y_, z_ = self.xyz
            for i, ax in enumerate(axes):
                mx, my, mz = self.rotation_bwd_m[i, :]
                new_ax = ne.evaluate(
                    "mx*(x_-x0) + my*(y_-y0) + mz*(z_-z0)", global_dict=self.__dict__
                )
                setattr(self, ax, new_ax)
        else:
            self.x, self.y, self.z = self.xyz

    def get_EB_out(self, E_out, B_out, mode):
        """
        Get the output arrays for the electric and magnetic fields.

        Parameters
        ----------
        E_out : array-like, optional
            Output array for the electric field.
        B_out : array-like, optional
            Output array for the magnetic field.
        mode : str
            Mode of calculation ('real' or 'complex').

        Returns
        -------
        tuple of array-like
            The output arrays for the electric and magnetic fields.
        """
        dtype = np.float64 if mode == "real" else np.complex128
        if E_out is None:
            E_out = [np.zeros(self.grid_shape, dtype=dtype) for _ in range(3)]
        if B_out is None:
            B_out = [np.zeros(self.grid_shape, dtype=dtype) for _ in range(3)]
        return E_out, B_out
    
    def rotate_fields_back(self, E_out, B_out, mode):
        """
        Rotate the fields back to the original coordinate frame.

        Parameters
        ----------
        E_out : array-like, optional
            Output array for the electric field.
        B_out : array-like, optional
            Output array for the magnetic field.
        mode : str
            Mode of calculation ('real' or 'complex').

        Returns
        -------
        tuple of array-like
            The rotated electric and magnetic fields.
        """
        E_out, B_out = self.get_EB_out(E_out, B_out, mode)

        out_dtype = config.FDTYPE if mode == "real" else config.CDTYPE
        # Transform to the original coordinate frame
        for i, (Ei, Bi) in enumerate(zip(E_out, B_out)):
            mx, my, mz = self.rotation_m[i, :]
            Ei += ne.evaluate("mx*Ex + my*Ey + mz*Ez", global_dict=self.__dict__).astype(out_dtype)
            Bi += ne.evaluate("mx*Bx + my*By + mz*Bz", global_dict=self.__dict__).astype(out_dtype)
        return E_out, B_out
    
    def convert_fields_to_real(self):
        """
        Take real part of field components.
        """
        for field in "Ex Ey Ez Bx By Bz".split():
            setattr(self, field, np.real(getattr(self, field)))


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
    """

    def __init__(self, field_params, grid):
        super().__init__(field_params, grid)

    def _allocate_fft(self):
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
        self._allocate_fft()
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
