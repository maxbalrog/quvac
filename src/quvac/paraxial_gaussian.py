'''This script implements analytic expression for paraxial gaussian'''
'''
TODO:
    - Add units to variables description
    - Test how transformed coordinates look like?
    - Check 0-order paraxial with plotting
'''

import numpy as np
from scipy.constants import pi, c, epsilon_0
from scipy.spatial.transform import Rotation


class ParaxialGaussianAnalytic(object):
    '''
    Analytic expression for paraxial Gaussian

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
    lmbd: float
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
                self.__dict__[key] = val * np.pi / 180
            else:
                self.__dict__[key] = val

        # Define grid variables
        self.grid = grid
        self.x_, self.y_, self.z_ = grid
        self.grid_shape = [dim.size for dim in grid]

        # Define additional field variables
        self.B0 = self.E0 / c
        self.k = 2. * pi / self.lmbd
        self.omega = c * self.k
        self.zR = pi * self.w0**2 / self.lmbd

        # Define rotation matrices and transform coordinate grid
        self.get_rotation()
        self.transform_coordinates()

        # Define variables not depending on time step
        self.w = self.w0 * np.sqrt(1 + (self.z/self.zR)**2)
        self.r2 = self.x**2 + self.y**2
        self.R = self.z + self.zR**2/self.z
        self.phase_no_t = self.phase0 - self.k*self.r2/(2*self.R) + np.arctan(self.z/self.zR)
        self.E = self.E0 * self.w0/self.w * np.exp(-self.r2/self.w**2)


    def get_rotation(self):
        # Define rotation transforming (0,0,1) -> (kx,ky,kz) for vectors
        self.rotation = Rotation.from_euler('zyz', (self.phi,self.theta,self.beta))
        self.rotation_m = self.rotation.as_matrix()
        # Inverse rotation: (kx,ky,kz) -> (0,0,1)
        self.rotation_bwd = self.rotation.inv()
        self.rotation_bwd_m = self.rotation_bwd.as_matrix()

    def transform_coordinates(self):
        axes = 'xyz'
        for i,ax in enumerate(axes):
            mx, my, mz = self.rotation_bwd_m[i]
            self.__dict__[ax] = mx*(self.x_-self.x0) + my*(self.y_-self.y0) + mz*(self.z_-self.z0)
    
    def calculate_field(self, t):
        psi_plane = self.w*(t-self.t0) - self.k*self.z
        phase = self.psi_no_t + psi_plane
        Ex = self.E * np.exp(-(psi_plane/self.omega)**2/(self.tau/2)**2) * np.cos(phase)
        return Ex
        


