'''
This script implements calculation of vacuum emission integral
It is planned to add support for two versions:
    - Calculation of total vacuum emission signal for given field configuration
    - Separation of fields into pump and probe with subsequent calculation of probe channel signal
'''
'''
TODO:
    - 
'''

import numpy as np
import numexpr as ne
from scipy.constants import pi, c, epsilon_0


class VacuumEmission(object):
    '''
    Calculator of Vacuum Emission amplitude from given fields

    Field parameters
    ----------------
    field: quvac.Field
        External fields
    '''
    def __init__(self, field):
        self.field = field

        angles = "theta phi beta".split()
        for angle in angles:
            self.__dict__[angle] = self.field.__dict__[angle]
        
        # Define two perpendicular polarizations
        self.e1 = np.array([np.cos(self.phi)*np.cos(self.theta),
                            np.sin(self.phi)*np.cos(self.theta),
                            -np.sin(self.theta)])
        self.e2 = np.array([-np.sin(self.phi),np.cos(self.phi),0])

        # Define symbolic expressions to evaluate later
        self.F = "0.5 * (Bx**2 + By**2 + Bz**2 - Ex**2 - Ey**2 - Ez**2)"
        self.G = "-(Ex*Bx + Ey*By + Ez*Bz)"
        self.U1 = [f"4*E{ax}*F + 7*B{ax}*G" for ax in "xyz"]
        self.U2 = [f"4*B{ax}*F + 7*E{ax}*G" for ax in "xyz"]
        self.I_ij = {f"{i}{j}": f"e{i}[0]*U{j}[0] + e{i}[1]*U{j}[1] + e{i}[2]*U{j}[2]"
                     for i in range(2) for j in range(2)}
        
    def calculate_one_time_step(self, t):
        pass

    def calculate_time_integral(self):
        pass

    def calculate_amplitude(self):
        pass

