'''This script implements analytic expression for paraxial gaussian'''
'''
TODO:
    - Add next order paraxial Gaussians (???)
    - Rewrite the main computation with numexpr
'''

import numpy as np
import numexpr as ne
from scipy.constants import pi, c
from scipy.spatial.transform import Rotation

from quvac.field.abc import AnalyticField
from quvac.field.utils import get_field_energy


class ParaxialGaussianAnalytic(AnalyticField):
    '''
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
    '''

    def __init__(self, field_params, grid):
        # Dynamically create class instance variables available with 
        # self.<variable_name>
        angles = 'theta phi beta phase0'.split()
        for key,val in field_params.items():
            if key in angles:
                val *= pi / 180
            self.__dict__[key] = val

        if 'E0' not in field_params:
            assert 'W' in field_params, """Field params need to have either 
                                           W (energy) or E0 (amplitude) as key"""
            self.E0 = 1.

        # Define grid variables
        self.grid = tuple(ax for ax in grid)
        self.grid_shape = tuple(ax.size for ax in grid)
        self.x_, self.y_, self.z_ = np.meshgrid(*grid, indexing='ij', sparse=True)
        self.dV = np.prod([ax[1]-ax[0] for ax in grid])

        # Define additional field variables
        self.x0, self.y0, self.z0 = self.focus_x
        self.t0 = self.focus_t
        self.B0 = self.E0 / c
        self.k = 2. * pi / self.lam
        self.omega = c * self.k
        self.zR = pi * self.w0**2 / self.lam

        # Rotate coordinate grid
        self.rotate_coordinates()

        # Define variables not depending on time step
        self.w = "(w0 * sqrt(1 + (z/zR)**2))"
        self.r2 = "(x**2 + y**2)"
        self.R = "(z + zR**2/z)"
        self.E_expr = f"B0 * w0/{self.w} * exp(-{self.r2}/{self.w}**2)"
        self.phase_no_t = ne.evaluate(f"phase0 - k*{self.r2}/(2*{self.R}) + arctan(z/zR)",
                                      global_dict=self.__dict__)
        self.E = ne.evaluate(self.E_expr, global_dict=self.__dict__)

        # Set up correct field amplitude
        if 'W' in field_params:
            self.check_energy()

    def get_rotation(self):
        # Define rotation transforming (0,0,1) -> (kx,ky,kz) for vectors
        # and (1,0,0) -> e(beta) = e1*cos(beta) + e2*sin(beta)
        self.rotation = Rotation.from_euler('ZYZ', (self.phi,self.theta,self.beta))
        self.rotation_m = self.rotation.as_matrix()
        # Inverse rotation: (kx,ky,kz) -> (0,0,1)
        self.rotation_bwd = self.rotation.inv()
        self.rotation_bwd_m = self.rotation_bwd.as_matrix()

    def rotate_coordinates(self):
        self.get_rotation()
        axes = 'xyz'
        for i,ax in enumerate(axes):
            mx, my, mz = self.rotation_bwd_m[i]
            self.__dict__[ax] = ne.evaluate("mx*(x_-x0) + my*(y_-y0) + mz*(z_-z0)",
                                            global_dict=self.__dict__)
            
    def check_energy(self):
        E, B = self.calculate_field(t=0)
        W = get_field_energy(E, B, self.dV)

        if 'W' in self.__dict__.keys() and not np.isclose(W, self.W, rtol=1e-5):
            self.E0 *= np.sqrt(self.W/W)
            self.B0 = self.E0 / c
            self.E = ne.evaluate(self.E_expr, global_dict=self.__dict__)
            self.W_num = W*self.E0**2

    
    def calculate_field(self, t, E_out=None, B_out=None):
        self.psi_plane = ne.evaluate("omega*(t-t0) - k*z", global_dict=self.__dict__)
        self.phase = "(phase_no_t + psi_plane)"
        Ex = ne.evaluate(f"E * exp(-(psi_plane/omega)**2/(tau/2)**2) * cos({self.phase})",
                         global_dict=self.__dict__)
        Ey, Ez = 0., 0.
        By = Ex
        Bx, Bz = 0., 0.
        if E_out is None:
            E_out = [np.zeros(self.grid_shape) for _ in range(3)]
            B_out = [np.zeros(self.grid_shape) for _ in range(3)]
        
        # Transform to the original coordinate frame
        for i,(Ei,Bi) in enumerate(zip(E_out,B_out)):
            mx, my, mz = self.rotation_m[i]
            ne.evaluate('Ei + mx*Ex + my*Ey + mz*Ez', out=Ei)
            ne.evaluate('Bi + mx*Bx + my*By + mz*Bz', out=Bi)

        return E_out, B_out
        


