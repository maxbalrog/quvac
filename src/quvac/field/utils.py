"""
Here we provide utility functions related to fields
"""

import numpy as np
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


def convert_tau(tau, mode="1/e^2"):
    '''
    This function converts between our duration (1/e^2) and
    FWHM and Standard Deviation ones. We assume the temporal
    envelope to have the form exp(-t**2/(tau/2)**2)
    '''
    match mode:
        case "tau":
            result = {'1/e^2': tau, 'FWHM': tau*np.sqrt(np.log(2)),
                      'FWHM-Intensity': tau*np.sqrt(np.log(2)/2),
                      'std': tau/(2*np.sqrt(2))}
        case "FWHM":
            result = {'1/e^2': tau/np.sqrt(np.log(2)), 'FWHM': tau,
                      'FWHM-Intensity': tau/np.sqrt(2),
                      'std': tau/(2*np.sqrt(2*np.log(2)))}
        case "FWHM-Intensity":
            result = {'1/e^2': tau/np.sqrt(np.log(2)/2), 'FWHM': tau*np.sqrt(2),
                      'FWHM-Intensity': tau, 'std': tau/(2*np.sqrt(np.log(2)))}
        case "std":
            result = {'1/e^2': 2*np.sqrt(2)*tau, 'FWHM': 2*tau*np.sqrt(2*np.log(2)),
                      'FWHM-Intensity': 2*tau*np.sqrt(np.log(2)), 'std': tau}
    result = {k: float(v) for k, v in result.items()}
    return result


def get_max_edge_intensity(E):
    """
    Calculate maximum fields at the edges of the grid
    """
    Ex, Ey, Ez = E
    nx, ny, nz = Ex.shape
    I = np.sqrt(abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2)
    edges = [np.max(I[0,:,:]), np.max(I[-1,:,:]),
             np.max(I[:,0,:]), np.max(I[:,-1,:]),
             np.max(I[:,:,0]), np.max(I[:,:,-1])]
    Iedge = np.max(edges)
    print(f"Max intensity at edges: {Iedge/I.max():.2e}")
    return Iedge


def get_intensity(field, t):
    E, B = field.calculate_field(t=t)
    E, B = [np.real(Ex) for Ex in E], [np.real(Bx) for Bx in B]
    I = (E[0]**2 + E[1]**2 + E[2]**2 + B[0]**2 + B[1]**2 + B[2]**2)/2
    return I