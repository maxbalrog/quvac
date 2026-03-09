"""
Basic linear Maxwell propagation class for a single field and 
a unified interface combining several Maxwell fields into one.
"""

import logging

import numexpr as ne
import numpy as np
import pyfftw
from scipy.constants import c, pi

from quvac import config
from quvac.field import SPATIAL_MODEL_FIELDS
from quvac.field.abc import Field
from quvac.pyfftw_executor import setup_fftw_executor

_logger = logging.getLogger("simulation")


class MaxwellField(Field):
    """
    Basic linear Maxwell propagation class for a single field.

    For such fields, the initial field distribution (spectral coefficients)
    at a certain time step is given with an analytic expression or from a file.
    For later time steps, the field is propagated according to linear Maxwell
    equations.

    Parameters
    ----------
    grid : quvac.grid.GridXYZ
        Spatial and spectral grid.
    fft_executor: quvac.pyfftw_executor.FFTExecutor, optional
        Executor that performs FFTs.
    nthreads : int, optional
        Number of threads to use for calculations. If not provided, defaults to 
        the number of CPU cores.
    """

    def __init__(self, grid, fft_executor=None, nthreads=None):
        self.grid_xyz = grid
        self.__dict__.update(self.grid_xyz.__dict__)

        self.nthreads = nthreads
        self.fft_executor = fft_executor

        self.c = c
        self.norm_ifft = self.dVk / (2.0 * pi) ** 3

        self.prefactor_expr = "exp(-1j*kabs*c*(t-t0)) * norm_ifft"

        self.E_expr = "prefactor * (e1*a1 + e2*a2)"
        self.B_expr = "prefactor * (e2*a1 - e1*a2)"

        self.EB_pairs = [
            [self.E_expr, None],
            [self.B_expr, None],
        ]

    def _allocate_ifft(self):
        """
        Allocate memory for inverse FFT calculations and define dictionaries
        for numexpr.evaluate().
        """
        self.fft_executor = setup_fftw_executor(self.fft_executor, self.vector_shape, 
                                                self.nthreads)
        
        self.prefactor = np.zeros(self.grid_shape, dtype=config.CDTYPE)

        self.prefactor_dict = {
            "kabs": self.kabs,
            "c": c,
            "t0": self.t0,
            "norm_ifft": self.norm_ifft,
        }

        self.EB_dict = {
            "e1": self.e1,
            "e2": self.e2,
            "a1": self.a1,
            "a2": self.a2,
            "prefactor": self.prefactor,
        }

    def calculate_field(self, t, E_out=None, B_out=None):
        """
        Calculates the electric and magnetic fields at a given time step.
        """
        if E_out is None:
            E_out = np.zeros(self.vector_shape, dtype=config.CDTYPE)
            B_out = np.zeros(self.vector_shape, dtype=config.CDTYPE)
        self.EB_pairs[0][1] = E_out
        self.EB_pairs[1][1] = B_out

        # Calculate prefactor at time t
        self.prefactor_dict.update({"t": t})
        ne.evaluate(self.prefactor_expr, local_dict=self.prefactor_dict, 
                    out=self.prefactor)

        # Calculate fourier of fields at time t and transform back to
        # spatial domain
        for expr,out_array in self.EB_pairs:  # noqa: B905
            ne.evaluate(expr, local_dict=self.EB_dict, out=self.fft_executor.tmp)
            self.fft_executor.backward_fftw.execute()
            np.copyto(out_array, self.fft_executor.tmp)
        
        return E_out, B_out


class MaxwellMultiple(MaxwellField):
    """
    Combine spectral coefficients from several fields and
    propagate them as one field.

    Parameters
    ----------
    fields : dict or list of dict
        Parameters of the fields.
    grid : quvac.grid.GridXYZ
        Spatial and spectral grid.
    fft_executor: quvac.pyfftw_executor.FFTExecutor, optional
        Executor that performs FFTs.
    nthreads : int, optional
        Number of threads to use for calculations. If not provided, defaults to the 
        number of CPU cores.

    Attributes
    ----------
    a1 : np.array
        First spectral coefficient.
    a2 : np.array
        Second spectral coefficient.
    fields : list of dict
        List of field parameter dictionaries.
    t0 : float
        Initial time for the field.
    """

    def __init__(self, fields, grid, fft_executor=None, nthreads=None):
        self.fft_executor = fft_executor
        super().__init__(grid, fft_executor, nthreads)

        self.a1, self.a2 = [
            pyfftw.zeros_aligned(self.grid_shape, dtype=config.CDTYPE) for _ in range(2)
        ]
        self.fields = [fields] if isinstance(fields, dict) else fields
        for i, field in enumerate(self.fields):
            _logger.info(f"Setting up field {i+1}:")
            a1, a2 = self.get_a12_from_field(field)
            self.a1 += a1
            self.a2 += a2

        self._allocate_ifft()

    def get_a12_from_field(self, field_params):
        """
        Get the spectral coefficients a1 and a2 from the field parameters.

        Parameters
        ----------
        field_params : dict
            Dictionary containing the parameters for the field.

        Returns
        -------
        tuple of np.array
            The spectral coefficients a1 and a2.

        Raises
        ------
        NotImplementedError
            If the field type is not supported.
        """
        field_type = field_params["field_type"]
        if field_type in SPATIAL_MODEL_FIELDS:
            cls = SPATIAL_MODEL_FIELDS[field_type]
            _logger.info(f"    {field_type}: {cls.__name__}")
            ini_field = cls(field_params, self.grid_xyz)
            self.t0 = ini_field.t0
            a1, a2 = ini_field.get_a12(ini_field.t0, self.fft_executor)
        else:
            raise NotImplementedError(
                f"`{field_type}` is not implemented, use"
                f"one of {list(SPATIAL_MODEL_FIELDS.keys())}"
            )
        return a1, a2

    def calculate_field(self, t, E_out=None, B_out=None):
        """
        Calculates the electric and magnetic fields at a given time step.
        """
        return super().calculate_field(t, E_out, B_out)
