"""
Here we provide utility functions related to fields
"""

import numexpr as ne
from scipy.constants import c, epsilon_0, mu_0, pi


def get_field_energy(E, B, dV):
    """
    Calculate numerical energy of field on spatial grid

    Parameters:
    -----------
    E, B: np.arrays
        Electric and magnetic fields
    dV: float
        Volume element
    """
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    W = 0.5 * epsilon_0 * c**2 * dV * ne.evaluate("sum(Ex**2 + Ey**2 + Ez**2)")
    W += 0.5 / mu_0 * dV * ne.evaluate("sum(Bx**2 + By**2 + Bz**2)")
    return W


def get_field_energy_kspace(a1, a2, k, dVk, mode="without 1/k"):
    """
    Calculate numerical energy of field on spectral grid

    Parameters:
    -----------
    a1, a2: np.arrays
        Spectral coefficients after projection
    k: np.array
        Wavevector
    dVk: float
        Spectral volume element
    mode: 'without 1/k' | 'with 1/k'
        Convention for the definition of a1,a2
        (with or without 1/k factor)
    """
    a = "(a1.real**2 + a1.imag**2 + a2.real**2 + a2.imag**2)"
    expr = f"sum({a})" if mode == "without 1/k" else f"sum(k**2 * {a})"
    W = 0.5 * epsilon_0 * c**2 * dVk / (2 * pi) ** 3 * ne.evaluate(expr)
    return W
