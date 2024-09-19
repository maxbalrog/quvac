'''
Here we provide utility functions related to fields
'''

import numpy as np
import numexpr as ne
from scipy.constants import pi, c, epsilon_0, mu_0


def get_field_energy(E, B, dV):
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    W = 0.5 * epsilon_0 * c**2 * dV * ne.evaluate('sum(Ex**2 + Ey**2 + Ez**2)')
    W += 0.5/mu_0 * dV * ne.evaluate('sum(Bx**2 + By**2 + Bz**2)')
    return W

