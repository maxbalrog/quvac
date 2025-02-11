"""
This script provides basic linear Maxwell propagation class
and a particular implementation of GaussianMaxwell
"""

import logging
import os

import numexpr as ne
import numpy as np
import pyfftw
from scipy.constants import c, pi

from quvac import config
from quvac.field.abc import Field
from quvac.field.dipole import DipoleAnalytic
from quvac.field.gaussian import GaussianAnalytic

SPATIAL_MODEL_FIELDS = {
    "dipole_maxwell": DipoleAnalytic,
    "paraxial_gaussian_maxwell": GaussianAnalytic,
}


logger = logging.getLogger("simulation")


class MaxwellField(Field):
    """
    For such fields the initial field distribution (spectral coefficients)
    at a certain time step is given with analytic expression or from file.
    For later time steps the field is propagated according to linear Maxwell
    equations

    Parameters:
    -----------
    grid: quvac.grid.GridXYZ
        spatial and spectral grid
    """

    def __init__(self, grid, nthreads=None):
        self.grid_xyz = grid
        self.__dict__.update(self.grid_xyz.__dict__)

        self.nthreads = nthreads if nthreads else os.cpu_count()

        self.c = c
        self.norm_ifft = self.dVk / (2.0 * pi) ** 3

        # 1st list for E, 2nd list for B
        self.EB_expr = [f"(e1{ax}*a1t + e2{ax}*a2t)" for ax in "xyz"] + [
            f"(e2{ax}*a1t - e1{ax}*a2t)" for ax in "xyz"
        ]

        self.allocate_tmp()

    def allocate_ifft(self):
        self.EB = [
            pyfftw.zeros_aligned(self.grid_shape, dtype=config.CDTYPE) for _ in range(6)
        ]
        self.a1t, self.a2t = [
            np.zeros(self.grid_shape, dtype="complex128") for _ in range(2)
        ]
        # pyfftw scheme
        self.EB_fftw = [
            pyfftw.FFTW(
                a,
                a,
                axes=(0, 1, 2),
                direction="FFTW_BACKWARD",
                flags=("FFTW_MEASURE",),
                threads=self.nthreads,
            )
            for a in self.EB
        ]
        # a = pyfftw.zeros_aligned(self.grid_shape, dtype=config.CDTYPE)
        # self.EB_fftw = pyfftw.FFTW(
        #                     a,
        #                     a,
        #                     axes=(0, 1, 2),
        #                     direction="FFTW_BACKWARD",
        #                     flags=("FFTW_MEASURE",),
        #                     threads=self.nthreads,
        #                )

        self.a_dict = {
            "kabs": self.kabs,
            "c": c,
            "t0": self.t0,
            "norm_ifft": self.norm_ifft,
            "a1": self.a1,
            "a2": self.a2,
        }

        self.EB_dict = {
            "e1x": self.e1x,
            "e1y": self.e1y,
            "e1z": self.e1z,
            "e2x": self.e2x,
            "e2y": self.e2y,
            "e2z": self.e2z,
            "a1t": self.a1t,
            "a2t": self.a2t,
        }

    def allocate_tmp(self):
        self.tmp = pyfftw.zeros_aligned(self.grid_shape, dtype="complex128")

    # def calculate_field(self, t, E_out=None, B_out=None):
    def calculate_field(self, t, E_out=None, B_out=None):
        if E_out is None:
            E_out = [np.zeros(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)]
            B_out = [np.zeros(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)]

        # Calculate a1,a2 at time t
        self.a_dict.update({"t": t})
        ne.evaluate(
            "exp(-1j*kabs*c*(t-t0)) * a1 * norm_ifft",
            local_dict=self.a_dict,
            out=self.a1t,
        )
        ne.evaluate(
            "exp(-1j*kabs*c*(t-t0)) * a2 * norm_ifft",
            local_dict=self.a_dict,
            out=self.a2t,
        )

        # Calculate fourier of fields at time t and transform back to
        # spatial domain
        for idx in range(6):
            ne.evaluate(self.EB_expr[idx], local_dict=self.EB_dict, out=self.EB[idx])
            if idx < 3:
                # E_out[idx][:] = self.tmp.astype(config.CDTYPE)
                # self.EB_fftw.update_arrays(E_out[idx], E_out[idx])
                self.EB_fftw[idx].execute()
                self.E_out[idx][:] = self.EB[idx].astype(config.CDTYPE)
            else:
                # B_out[idx-3][:] = self.tmp.astype(config.CDTYPE)
                # self.EB_fftw.update_arrays(B_out[idx-3], B_out[idx-3])
                self.EB_fftw[idx].execute()
                self.B_out[idx-3][:] = self.EB[idx].astype(config.CDTYPE)
            # self.EB_fftw.execute()
            # ne.evaluate(self.EB_expr[idx], local_dict=self.EB_dict, out=self.tmp)
            # if idx < 3:
            #     E_out[idx][:] = self.tmp.astype(config.CDTYPE)
            #     self.EB_fftw.update_arrays(E_out[idx], E_out[idx])
            # else:
            #     B_out[idx-3][:] = self.tmp.astype(config.CDTYPE)
            #     self.EB_fftw.update_arrays(B_out[idx-3], B_out[idx-3])
            # self.EB_fftw.execute()
        return E_out, B_out


class MaxwellMultiple(MaxwellField):
    """
    Combine spectral coefficients from several fields and
    propagate them as one field

    Parameters:
    -----------
    fields: dict | list of dicts
        Parameters of fields
    grid: quvac.grid.GridXYZ
        spatial and spectral grid
    nthreads: int (optional)
        number of threads to use for pyfftw
    """

    def __init__(self, fields, grid, nthreads=None):
        super().__init__(grid, nthreads)

        self.a1, self.a2 = [
            pyfftw.zeros_aligned(self.grid_shape, dtype=config.CDTYPE) for _ in range(2)
        ]
        self.fields = [fields] if isinstance(fields, dict) else fields
        for i, field in enumerate(self.fields):
            logger.info(f"Setting up field {i+1}:")
            a1, a2 = self.get_a12_from_field(field)
            self.a1 += a1
            self.a2 += a2

        self.allocate_ifft()

    def get_a12_from_field(self, field_params):
        field_type = field_params["field_type"]
        if field_type in SPATIAL_MODEL_FIELDS:
            cls = SPATIAL_MODEL_FIELDS[field_type]
            logger.info(f"    {field_type}: {cls.__name__}")
            ini_field = cls(field_params, self.grid_xyz)
            self.t0 = ini_field.t0
            a1, a2 = ini_field.get_a12(ini_field.t0)
        else:
            raise NotImplementedError(
                f"`{field_type}` is not implemented, use"
                f"one of {list(SPATIAL_MODEL_FIELDS.keys())}"
            )
        return a1, a2

    def calculate_field(self, t, E_out=None, B_out=None):
        return super().calculate_field(t, E_out, B_out)
