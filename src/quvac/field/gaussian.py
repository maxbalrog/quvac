"""
Analytic expression for paraxial gaussian (0-order and higher-orders).

----

.. [1] Y. I. Salamin. "Fields of a Gaussian beam beyond the paraxial 
    approximation." Applied Physics B 86 (2007): 319-326.
"""

import numexpr as ne
import numpy as np
from scipy.constants import c, pi

from quvac.field.abc import ExplicitField


class GaussianAnalytic(ExplicitField):
    """
    Analytic expression for paraxial Gaussian beam.

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
            - 'E0' : float, optional
                Amplitude (either E0 or W is required).
            - 'W' : float, optional
                Energy (either E0 or W is required).
    grid : quvac.grid.GridXYZ
        Spatial and grid.

    Notes
    -----
    All field parameters are in SI units.

    Higher-order paraxial Gaussian orders are from [1]_.
    """

    def __init__(self, field_params, grid):
        super().__init__(field_params, grid)

        self.phase0 += pi / 2.0
        self.order = getattr(self, "order", 0)

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

        # Rotate coordinate grid
        self.rotate_coordinates()

        # Define variables not depending on time step
        self.w = "(w0 * sqrt(1. + (z/zR)**2))"
        self.r2 = "(x**2 + y**2)"
        self.R = "(z + zR**2/z)"
        self.E_expr = f"B0 * w0/{self.w} * exp(-{self.r2}/{self.w}**2)"
        self.phase_no_t = ne.evaluate(
            f"phase0 - k*{self.r2}/(2.*{self.R}) + arctan(z/zR)",
            global_dict=self.__dict__,
        )

        self.E = ne.evaluate(self.E_expr, global_dict=self.__dict__)

        if self.order > 0:
            self.calculate_ho_orders()

        # Set up correct field amplitude
        if "W" in field_params:
            self.check_energy()

    def define_ho_variables(self):
        """
        Define higher order variables.
        """
        self.eps = self.w0 / self.zR
        self.xi = ne.evaluate("x/w0", global_dict=self.__dict__)
        self.nu = ne.evaluate("y/w0", global_dict=self.__dict__)
        self.zeta = ne.evaluate("z/zR", global_dict=self.__dict__)
        self.rho = ne.evaluate("sqrt(xi**2 + nu**2)", global_dict=self.__dict__)
        self.f = ne.evaluate(
            "exp(-1j*arctan(zeta))/sqrt(1 + zeta**2)", global_dict=self.__dict__
        )
        for n in range(1, 5):
            self.__dict__[f"f{n}"] = ne.evaluate(
                "1/(1+zeta**2)**(n/2) * exp(-1j*n*arctan(zeta))",
                global_dict=self.__dict__,
            )

        self.Ex_terms = {
            0: "1",
            2: "eps**2 * f2 * (xi**2 - f1*rho**4/4)",
            4: (
                "eps**4 * f2 * (1/8 - f1*rho**2/4 + f2*rho**2*(xi**2 - rho**2/16) - "
                "f3*rho**4*(xi**2/4 + rho**2/8) + f4*rho**8/32)"
            ),
        }
        self.Ey_terms = {
            0: "0",
            2: "eps**2 * f2",
            4: "eps**4*f4*rho**2 * (1 - f1*rho**2/4)",
        }
        self.Ez_terms = {
            0: "0",
            1: "eps * f1",
            3: "eps**3 * f2 * (-0.5 + f1*rho**2 - f2*rho**4/4)",
            5: (
                "eps**5 * f3 * (-3/8 - 3*f1*rho**2/8 + 17*f2*rho**4/16 - "
                "3*f3*rho**6/8 + f4*rho**8/32)"
            ),
        }
        self.By_terms = {
            0: "1",
            2: "eps**2 * f2*rho**2 * (1/2 - f1*rho**2/4)",
            4: "eps**4 * f2 * (-1/8 + f1*rho**2/4 + 5*f2*rho**4/16 - f3*rho**6/4 + f4*rho**8/32)",
        }
        self.Bz_terms = {
            0: "0",
            1: "eps * f1",
            3: "eps**3*f2 * (1/2 + f1*rho**2/2 - f2*rho**4/4)",
            5: "eps**5*f3 * (3/8 + 3*f1*rho**2/8 + 3*f2*rho**4/16 - f3*rho**6/4 + f4*rho**8/32)",
        }

    def calculate_ho_orders(self):
        """
        Calculate higher order terms for the fields.
        """
        self.define_ho_variables()
        # For a given order, combine final expression to calculate
        names = "Ex Ey Ez By Bz".split()
        for name in names:
            terms = self.__dict__[f"{name}_terms"]
            self.__dict__[f"{name}_expr"] = " + ".join(
                [term for order, term in terms.items() if order <= self.order]
            )
            self.__dict__[f"{name}_ho"] = ne.evaluate(
                self.__dict__[f"{name}_expr"], global_dict=self.__dict__
            )

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
        k = 2.0 * pi / self.lam
        self.psi_plane = ne.evaluate("(omega*(t-t0) - k*z)", global_dict=self.__dict__)
        self.phase = "(phase_no_t + psi_plane)"

        Et = ne.evaluate(
            f"E * exp(-(2.*psi_plane/(omega*tau))**2) * exp(-1.j*{self.phase})",
            global_dict=self.__dict__,
        )

        if self.order > 0:
            self.Ex = ne.evaluate("1j*Et * Ex_ho", global_dict=self.__dict__)
            self.Ey = ne.evaluate("1j*Et * Ey_ho * xi * nu", global_dict=self.__dict__)
            self.Ez = ne.evaluate("Et * Ez_ho * xi", global_dict=self.__dict__)
            self.Bx = 0.0
            self.By = ne.evaluate("1j*Et * By_ho", global_dict=self.__dict__)
            self.Bz = ne.evaluate("Et * Bz_ho * nu", global_dict=self.__dict__)
        else:
            self.Ex = self.By = 1j * Et.copy()
            self.Ey = self.Ez = self.Bx = self.Bz = 0.0

        if mode == "real":
            self.convert_fields_to_real()

        E_out, B_out = self.rotate_fields_back(E_out, B_out, mode)
        return E_out, B_out
