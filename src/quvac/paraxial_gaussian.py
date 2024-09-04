'''This script implements analytic expression for paraxial gaussian'''
'''
TODO:
    - Add units to variables description
'''

import numpy as np
from scipy.constants import pi, c, epsilon_0


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
        for key,val in field_params.items():
            self.__dict__[key] = val

        # Define grid variables
        self.x, self.y, self.z = grid
        self.grid_shape = [dim.size for dim in grid]

        # Define additional field variables
        self.B0 = self.E0 / c
        self.k = 2. * pi / self.lmbd
        self.omega = c * self.k
        self.zR = pi * self.w0**2 / self.lmbd

