"""
Analytic expression for Laguerre-Gaussian modes.

----

.. [1] L. Allen et al. "Orbital angular momentum of light and the transformation of 
Laguerre-Gaussian laser modes." PRA 45.11 (1992): 8185.
"""
from math import factorial

import numexpr as ne
import numpy as np
from scipy.constants import c, pi
from scipy.special import genlaguerre

from quvac.field.abc import ExplicitField


class LaguerreGaussianAnalytic(ExplicitField):
    """
    Analytic expression for Laguerre-Gaussian modes.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters. Required keys are:
        - 'focus_x' : tuple of float
            Location of spatial focus (x, y, z).
        - 'focus_t' : float
            Location of temporal focus.
        - 'theta' : float
            Polar angle of k-vector (in degrees).
        - 'phi' : float
            Azimuthal angle of k-vector (in degrees).
        - 'beta' : float
            Polarization angle (in degrees).
        - 'lam' : float
            Wavelength of the pulse.
        - 'w0' : float
            Waist size.
        - 'tau' : float
            Duration.
        - 'phase0' : float
            Phase delay at focus.
        - 'p' : int
            Radial index of the Laguerre-Gaussian mode.
        - 'l' : int
            Azimuthal index of the Laguerre-Gaussian mode.
        - 'E0' : float, optional
            Amplitude (either E0 or W is required).
        - 'W' : float, optional
            Energy (either E0 or W is required).
    grid : quvac.grid.GridXYZ
        Spatial and spectral grid.

    Notes
    -----
    All field parameters are in SI units.

    Analytic expression is from [1]_.
    """

    def __init__(self, field_params, grid):
        super().__init__(field_params, grid)

        self.phase0 += pi / 2.0

        if "E0" not in field_params:
            err_msg = ("Field params need to have either W (energy) or"
                       "E0 (amplitude) as key")
            assert "W" in field_params, err_msg
            self.E0 = 1.0e10

        # Define additional field variables
        self.x0, self.y0, self.z0 = self.focus_x
        self.t0 = self.focus_t
        self.B0 = self.E0 / c
        self.k = 2.0 * pi / self.lam
        self.omega = c * self.k
        self.zR = pi * self.w0**2 / self.lam
        self.prefactor = np.sqrt(2*factorial(self.p)/factorial(self.p+self.l)/pi)

        # Rotate coordinate grid
        self.rotate_coordinates()

        # Define variables not depending on time step
        self.w = "(w0 * sqrt(1. + (z/zR)**2))"
        self.r = "sqrt(x**2 + y**2)"
        self.r2 = "(x**2 + y**2)"
        self.R_inv = "(z/(z**2 + zR**2))"
        
        self.rw = ne.evaluate(f"({self.r}*sqrt(2)/{self.w})", global_dict=self.__dict__)
        self.lag_poly = self.prefactor * genlaguerre(self.p, self.l)(self.rw**2)

        self.E_expr = (f"B0 * w0/{self.w} * exp(-{self.r2}/{self.w}**2) * "
                       f"rw**l * lag_poly")
        self.phase_no_t = ne.evaluate(
            (f"phase0 - k*{self.r2}*{self.R_inv}/2 + (2*p+l+1)*arctan(z/zR) - "
             "l*arctan2(y,x)"),
            global_dict=self.__dict__,
        )

        self.E = ne.evaluate(self.E_expr, global_dict=self.__dict__)

        # Set up correct field amplitude
        if "W" in field_params:
            self.check_energy()

    def check_energy(self):
        """
        Check and adjust the field energy.
        """
        self._check_energy()

        if self.modify_energy:
            self.E0 *= self.W_correction
            self.B0 = self.E0 / c
            self.E = ne.evaluate(self.E_expr, global_dict=self.__dict__)

    def calculate_field(self, t, E_out=None, B_out=None, mode="real"):
        """
        Calculates the electric and magnetic fields at a given time step.
        """
        k = 2.0 * pi / self.lam # noqa: F841
        self.psi_plane = ne.evaluate("(omega*(t-t0) - k*z)", global_dict=self.__dict__)
        self.phase = "(phase_no_t + psi_plane)"

        Et = ne.evaluate(
            f"E * exp(-(2.*psi_plane/(omega*tau))**2) * exp(-1.j*{self.phase})",
            global_dict=self.__dict__,
        )

        self.Ex = self.By = 1j * Et.copy()
        self.Ey = self.Ez = self.Bx = self.Bz = 0.0

        if mode == "real":
            self.convert_fields_to_real()

        E_out, B_out = self.rotate_fields_back(E_out, B_out, mode)
        return E_out, B_out
