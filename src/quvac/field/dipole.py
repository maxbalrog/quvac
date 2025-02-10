"""This script implements analytic expression for dipole wave"""

import numexpr as ne
import numpy as np
from scipy.constants import c, pi
from scipy.spatial.transform import Rotation

from quvac.field.abc import ExplicitField
from quvac.field.utils import get_field_energy


class DipoleAnalytic(ExplicitField):
    '''
    Dipole wave from Gonoskov, Ivan, et al. "Dipole pulse theory: 
    Maximizing the field amplitude from 4 π focused laser pulses." 
    PRA 86.5 (2012): 053836.

    d0 is along ez

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
    tau: float
        Duration
    W: float (optional)
        Energy
    '''
    def __init__(self, field_params, grid):
        angles = "theta phi beta".split()
        for key, val in field_params.items():
            if key in angles:
                val *= pi / 180.0
            self.__dict__[key] = val
        
        if "envelope" not in self.__dict__:
            self.envelope = "plane"

        # Define grid variables
        self.grid_xyz = grid
        grid_keys = "grid_shape xyz dV".split()
        self.__dict__.update(
            {k: v for k, v in self.grid_xyz.__dict__.items() if k in grid_keys}
        )

        # Define additional field variables
        self.x0, self.y0, self.z0 = self.focus_x
        self.t0 = self.focus_t
        # self.B0 = self.E0 / c
        self.k = 2.0 * pi / self.lam
        self.omega = c * self.k
        self.d0 = 1.

        # Rotate coordinate grid
        self.rotate_coordinates()

        # Define variables independent of time
        self.R_expr = 'sqrt(x**2 + y**2 + z**2)' # radius
        self.R =  ne.evaluate(self.R_expr, global_dict=self.__dict__)
        
        self.nx, self.ny, self.nz = [np.nan_to_num(ax/self.R) for ax in (self.x, self.y, self.z)]

        self.EB_dict = {"R": self.R,
                        "c": c,
                        "k": self.k,
                        "omega": self.omega,
                        "d0": self.d0,
                        "nx": self.nx,
                        "ny": self.ny,
                        "nz": self.nz}

        # Define envelope expressions
        if self.envelope == "plane":
            self.g_expr = "1j*exp(-1j*omega*t)"
            self.gdot_expr = 'omega*exp(-1j*omega*t)'
            self.gdotdot_expr = "-1j*omega**2*exp(-1j*omega*t)"
        elif self.envelope == "gauss":
            self.g_expr = '1j*exp(-(t/(tau/2))**2 - 1j*omega*t)'
            self.gdot_expr = '1j*exp(-(t/(tau/2))**2 - 1j*omega*t) * (-2*t/(tau/2)**2 - 1j*omega)'
            self.gdotdot_expr = ('1j*exp(-(t/(tau/2))**2 - 1j*omega*t) * (4*t**2/(tau/2)**4 '
                                 '- 2/(tau/2)**2 - omega**2 + 4j*t*omega/(tau/2)**2)')
            
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

    def check_energy(self):
        E, B = self.calculate_field(t=0)
        W = get_field_energy(E, B, self.dV)

        if "W" in self.__dict__.keys() and not np.isclose(W, self.W, rtol=1e-5):
            self.d0 *= np.sqrt(self.W / W)
            self.W_num = W * self.d0**2
        #     self.E0 *= np.sqrt(self.W / W)
        #     self.B0 = self.E0 / c
        #     self.E = ne.evaluate(self.E_expr, global_dict=self.__dict__)
        #     self.W_num = W * self.E0**2

    def g(self, t):
        return ne.evaluate(self.g_expr, global_dict=self.__dict__)
    
    def gdot(self, t):
        return ne.evaluate(self.gdot_expr, global_dict=self.__dict__)
    
    def gdotdot(self, t):
        return ne.evaluate(self.gdotdot_expr, global_dict=self.__dict__)
    
    def g_plusminus(self, t, sign=1):
        return self.g(t-self.R/c) + sign*self.g(t+self.R/c)
    
    def gdot_plusminus(self, t, sign=1):
        return self.gdot(t-self.R/c) + sign*self.gdot(t+self.R/c)
    
    def gdotdot_plusminus(self, t, sign=1):
        return self.gdotdot(t-self.R/c) + sign*self.gdotdot(t+self.R/c)

    def _fix_singularity(self, t):
        # fix divergence at R=0
        Nx,Ny,Nz = self.Ex.shape
        self.Ex[Nx//2,Ny//2,Nz//2] = 0.
        self.Ey[Nx//2,Ny//2,Nz//2] = 0.
        self.Ez[Nx//2,Ny//2,Nz//2] = 4/3*self.k**3*(1+6/(self.tau*self.omega)**2)*np.cos(self.omega*t)

        self.Bx[Nx//2,Ny//2,Nz//2] = 0.
        self.By[Nx//2,Ny//2,Nz//2] = 0.
    
    def calculate_field(self, t, E_out=None, B_out=None, mode="real"):
        gdotdot_p = self.gdotdot_plusminus(t)
        gdotdot_m = self.gdotdot_plusminus(t, sign=-1)
        gdot_p = self.gdot_plusminus(t)
        gdot_m = self.gdot_plusminus(t, sign=-1)
        g_m = self.g_plusminus(t, sign=-1)

        Bt = ne.evaluate("gdotdot_p/(R*c**2) + gdot_m/(R**2*c)", global_dict=self.EB_dict)
        Et = ne.evaluate("gdot_p/(c*R**2) + g_m/R**3", global_dict=self.EB_dict)
        
        self.Ex = ne.evaluate('nx*nz*gdotdot_m/(R*c**2) + 3*nx*nz*Et', global_dict=self.EB_dict)
        self.Ey = ne.evaluate('ny*nz*gdotdot_m/(R*c**2) + 3*ny*nz*Et', global_dict=self.EB_dict)
        self.Ez = ne.evaluate('-(nx**2+ny**2)*gdotdot_m/(R*c**2) + (3*nz**2-1)*Et', global_dict=self.EB_dict)

        self.Bx = ne.evaluate("-ny*Bt", global_dict=self.EB_dict)
        self.By = ne.evaluate("nx*Bt", global_dict=self.EB_dict)
        self.Bz = 0.
        
        # fix divergence at R=0
        self._fix_singularity(t)

        for field in "Ex Ey Ez Bx By Bz".split():
            self.__dict__[field] *= self.d0

        if mode == "real":
            for field in "Ex Ey Ez Bx By Bz".split():
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