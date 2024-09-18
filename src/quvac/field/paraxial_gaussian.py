'''This script implements analytic expression for paraxial gaussian'''
'''
TODO:
    - Add next order paraxial Gaussians (???)
    - Rewrite the main computation with numexpr
'''

import numpy as np
import numexpr as ne
from scipy.constants import pi, c, epsilon_0, mu_0
from scipy.spatial.transform import Rotation

from quvac.field.abc import AnalyticField


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
    E0: float
        Amplitude
    phase0: float
        Phase delay at focus 
    '''

    def __init__(self, field_params, grid):
        # Dynamically create class instance variables available with 
        # self.<variable_name>
        angles = 'theta phi beta phase0'.split()
        for key,val in field_params.items():
            if key in angles:
                self.__dict__[key] = val * pi / 180
            else:
                self.__dict__[key] = val
        if 'E0' not in field_params:
            assert 'W' in field_params, """Field params need to have either 
                                           W (energy) or E0 (amplitude) as key"""
            self.E0 = 1.

        # Define grid variables
        self.grid = [ax.flatten() for ax in grid]
        self.x_, self.y_, self.z_ = np.meshgrid(*grid, indexing='ij', sparse=True)
        self.grid_shape = [dim.size for dim in grid]
        self.dV = np.prod([ax[1]-ax[0] for ax in self.grid])

        # Define additional field variables
        self.x0, self.y0, self.z0 = self.focus_x
        self.t0 = self.focus_t
        self.B0 = self.E0 / c
        self.k = 2. * pi / self.lam
        self.omega = c * self.k
        self.zR = pi * self.w0**2 / self.lam

        # Define rotation matrices and transform coordinate grid
        self.get_rotation()
        self.transform_coordinates()

        # Define variables not depending on time step
        # self.w = self.w0 * np.sqrt(1 + (self.z/self.zR)**2)
        # self.r2 = self.x**2 + self.y**2
        # self.R = self.z + self.zR**2/self.z
        # self.phase_no_t = self.phase0 - self.k*self.r2/(2*self.R) + np.arctan(self.z/self.zR)
        # self.E = self.E0 * self.w0/self.w * np.exp(-self.r2/self.w**2)
        self.w = ne.evaluate("w0 * sqrt(1 + (z/zR)**2)", global_dict=self.__dict__)
        self.r2 = ne.evaluate("x**2 + y**2", global_dict=self.__dict__)
        self.R = ne.evaluate("z + zR**2/z", global_dict=self.__dict__)
        self.phase_no_t = ne.evaluate("phase0 - k*r2/(2*R) + arctan(z/zR)", global_dict=self.__dict__)
        # self.E = ne.evaluate("E0 * w0/w * exp(-r2/w**2)", global_dict=self.__dict__)
        self.E = ne.evaluate("B0 * w0/w * exp(-r2/w**2)", global_dict=self.__dict__)

        if 'W' in field_params:
            self.check_energy()
        self.check_energy()

    def get_rotation(self):
        # Define rotation transforming (0,0,1) -> (kx,ky,kz) for vectors
        self.rotation = Rotation.from_euler('ZYZ', (self.phi,self.theta,self.beta))
        self.rotation_m = self.rotation.as_matrix()
        # Inverse rotation: (kx,ky,kz) -> (0,0,1)
        self.rotation_bwd = self.rotation.inv()
        self.rotation_bwd_m = self.rotation_bwd.as_matrix()

    def transform_coordinates(self):
        axes = 'xyz'
        for i,ax in enumerate(axes):
            self.mx, self.my, self.mz = self.rotation_bwd_m[i]
            self.__dict__[ax] = ne.evaluate("mx*(x_-x0) + my*(y_-y0) + mz*(z_-z0)",
                                            local_dict=self.__dict__)
            
    def check_energy(self):
        E, B = self.calculate_field(t=0)
        Ex, Ey, Ez = E
        Bx, By, Bz = B
        W = 0.5 * epsilon_0 * c**2 * self.dV * ne.evaluate('sum(Ex**2 + Ey**2 + Ez**2)')
        W += 0.5/mu_0 * self.dV * ne.evaluate('sum(Bx**2 + By**2 + Bz**2)')

        if 'W' in self.__dict__.keys() and not np.isclose(W, self.W, rtol=1e-5):
            E_new = self.E0 * np.sqrt(self.W/W)
            self.E0 = E_new
            self.B0 = self.E0 / c
            self.E = ne.evaluate("B0 * w0/w * exp(-r2/w**2)", global_dict=self.__dict__)
        else:
            self.W_num = W

    
    def calculate_field(self, t, E_out=None, B_out=None):
        self.psi_plane = ne.evaluate("omega*(t-t0) - k*z", global_dict=self.__dict__)
        self.phase = ne.evaluate("phase_no_t + psi_plane", global_dict=self.__dict__)
        Ex = ne.evaluate("E * exp(-(psi_plane/omega)**2/(tau/2)**2) * cos(phase)",
                         global_dict=self.__dict__)
        Ey, Ez = 0., 0.
        By = Ex
        Bx, Bz = 0., 0.

        # Transform to the original coordinate frame
        if E_out is None:
            E_out = [np.zeros(self.grid_shape) for _ in range(3)]
        if B_out is None:
            B_out = [np.zeros(self.grid_shape) for _ in range(3)]
        
        for i,Ei in enumerate(E_out):
            mx, my, mz = self.rotation_m[i]
            ne.evaluate('Ei + mx*Ex + my*Ey + mz*Ez', out=Ei)

        for i,Bi in enumerate(B_out):
            mx, my, mz = self.rotation_m[i]
            ne.evaluate('Bi + mx*Bx + my*By + mz*Bz', out=Bi)
        return E_out, B_out
        


