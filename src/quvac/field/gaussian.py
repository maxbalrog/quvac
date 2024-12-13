"""This script implements analytic expression for paraxial gaussian"""

import numexpr as ne
import numpy as np
from scipy.constants import c, pi
from scipy.spatial.transform import Rotation

from quvac.field.abc import ExplicitField
from quvac.field.utils import get_field_energy


class GaussianAnalytic(ExplicitField):
    """
    Analytic expression for paraxial Gaussian
    All field parameters are in SI units

    Field parameters
    ----------------
    focus_x: (float, float, float)
        Location of spatial focus (x,y,z)
    focus_t: float
        Location of temporal focus
    theta, phi: float
        Spherical angles of k-vector (in degrees),
        theta - angle with z-axis,
        phi - angle with x-axis
    beta: float
        Polarization angle (in degrees),
        beta = 0, theta = 0 corresponds to E-vector along x-axis
    lam: float
        Lambda, pulse wavelength
    w0: float
        Waist size
    tau: float
        Duration
    phase0: float
        Phase delay at focus
    E0: float (optional, either E0 or W is required)
        Amplitude
    W: float (optional)
        Energy

    Other arguments
    ---------------
    grid: (1d-np.array, 1d-np.array, 1d-np.array)
        xyz spatial grid to calculate fields on
    """

    def __init__(self, field_params, grid):
        super().__init__(grid)
        # Dynamically create class instance variables available with
        # self.<variable_name>
        angles = "theta phi beta phase0".split()
        for key, val in field_params.items():
            if key in angles:
                val *= pi / 180.0
            self.__dict__[key] = val
        self.phase0 += pi / 2.0
        if "order" not in field_params:
            self.order = 0

        if "E0" not in field_params:
            assert (
                "W" in field_params
            ), """Field params need to have either 
                                           W (energy) or E0 (amplitude) as key"""
            self.E0 = 1.0e10

        # Define grid variables
        self.grid_xyz = grid
        grid_keys = "grid_shape xyz dV".split()
        self.__dict__.update(
            {k: v for k, v in self.grid_xyz.__dict__.items() if k in grid_keys}
        )

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

    def get_rotation(self):
        # Define rotation transforming (0,0,1) -> (kx,ky,kz) for vectors
        # and (1,0,0) -> e(beta) = e1*cos(beta) + e2*sin(beta)
        self.rotation = Rotation.from_euler("ZYZ", (self.phi, self.theta, self.beta))
        self.rotation_m = self.rotation.as_matrix()
        # Inverse rotation: (kx,ky,kz) -> (0,0,1)
        self.rotation_bwd = self.rotation.inv()
        self.rotation_bwd_m = self.rotation_bwd.as_matrix()

    def rotate_coordinates(self):
        self.get_rotation()
        axes = "xyz"
        x_, y_, z_ = self.xyz
        for i, ax in enumerate(axes):
            mx, my, mz = self.rotation_bwd_m[i, :]
            self.__dict__[ax] = ne.evaluate(
                "mx*(x_-x0) + my*(y_-y0) + mz*(z_-z0)", global_dict=self.__dict__
            )

    def define_ho_variables(self):
        """
        We follow the article: Salamin, Yousef I. "Fields of a Gaussian beam
        beyond the paraxial approximation." Applied Physics B 86 (2007): 319-326.
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
        ho stands for Higher Order
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
        E, B = self.calculate_field(t=0)
        W = get_field_energy(E, B, self.dV)

        if "W" in self.__dict__.keys() and not np.isclose(W, self.W, rtol=1e-5):
            self.E0 *= np.sqrt(self.W / W)
            self.B0 = self.E0 / c
            self.E = ne.evaluate(self.E_expr, global_dict=self.__dict__)
            self.W_num = W * self.E0**2

    def calculate_field(self, t, E_out=None, B_out=None, mode="real"):
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
            for field in "Ex Ey Ez By Bz".split():
                self.__dict__[field] = np.real(self.__dict__[field])

        dtype = np.float64 if mode == "real" else np.complex128
        if E_out is None:
            E_out = [np.zeros(self.grid_shape, dtype=dtype) for _ in range(3)]
        if B_out is None:
            B_out = [np.zeros(self.grid_shape, dtype=dtype) for _ in range(3)]

        # Transform to the original coordinate frame
        for i, (Ei, Bi) in enumerate(zip(E_out, B_out)):
            mx, my, mz = self.rotation_m[i, :]
            ne.evaluate("Ei + mx*Ex + my*Ey + mz*Ez", out=Ei, global_dict=self.__dict__)
            ne.evaluate("Bi + mx*Bx + my*By + mz*Bz", out=Bi, global_dict=self.__dict__)

        return E_out, B_out
