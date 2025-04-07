"""
Utility functions related to fields.
"""

import numexpr as ne
import numpy as np
from scipy.constants import c, epsilon_0, mu_0, pi


def get_field_energy(E, B, dV):
    """
    Calculate the numerical energy of the field on a spatial grid.

    Parameters
    ----------
    E : np.array
        Electric field components as a tuple of three numpy arrays (Ex, Ey, Ez).
    B : np.array
        Magnetic field components as a tuple of three numpy arrays (Bx, By, Bz).
    dV : float
        Volume element.

    Returns
    -------
    float
        The calculated energy of the field.

    Notes
    -----
    The energy is calculated using the formula:

    W = sum(0.5 * epsilon_0 * c**2 * dV * abs(E)**2 + 0.5 / mu_0 * dV * abs(B)**2)
    """
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    W = 0.5 * epsilon_0 * c**2 * dV * ne.evaluate("sum(Ex**2 + Ey**2 + Ez**2)")
    W += 0.5 / mu_0 * dV * ne.evaluate("sum(Bx**2 + By**2 + Bz**2)")
    return W


def get_field_energy_kspace(a1, a2, k, dVk, mode="without 1/k"):
    """
    Calculate the numerical energy of the field on a spectral grid.

    Parameters
    ----------
    a1 : np.array
        First spectral coefficient.
    a2 : np.array
        Second spectral coefficient.
    k : np.array
        Wavevector.
    dVk : float
        Spectral volume element.
    mode : str, optional
        Convention for the definition of a1 and a2 ('without 1/k' or 'with 1/k').
        Default is 'without 1/k'.

    Returns
    -------
    float
        The calculated energy of the field.

    Notes
    -----
    The energy is calculated using the formula:

    W = 0.5 * epsilon_0 * c**2 * dVk / (2 * pi)**3 * sum(abs(a1)**2 + abs(a2)**2)
    
    If mode is 'with 1/k', the formula is:
    
    W = 0.5 * epsilon_0 * c**2 * dVk / (2 * pi)**3 * sum(k**2 * {abs(a1)**2 + 
    abs(a2)**2})
    """
    a = "(a1.real**2 + a1.imag**2 + a2.real**2 + a2.imag**2)"
    expr = f"sum({a})" if mode == "without 1/k" else f"sum(k**2 * {a})"
    W = 0.5 * epsilon_0 * c**2 * dVk / (2 * pi) ** 3 * ne.evaluate(expr)
    return W


def convert_tau(tau, mode="1/e^2"):
    """
    Convert between different pulse duration conventions.

    Parameters
    ----------
    tau : float
        Pulse duration.
    mode : str, optional
        The convention of the input pulse duration. Options are '1/e^2',
        'FWHM', 'FWHM-Intensity', and 'std'. Default is '1/e^2'.

    Returns
    -------
    dict
        Dictionary containing the pulse duration in different conventions:
            - '1/e^2' : float
                Pulse duration in 1/e^2 convention.
            - 'FWHM' : float
                Pulse duration in Full Width at Half Maximum (FWHM) convention.
            - 'FWHM-Intensity' : float
                Pulse duration in FWHM-Intensity convention.
            - 'std' : float
                Pulse duration in standard deviation convention.

    Notes
    -----
    The temporal envelope is assumed to have the form exp(-t**2/(tau/2)**2).
    """
    match mode:
        case "1/e^2":
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
        case _:
            raise NotImplementedError(f"Mode {mode} is not supported.")
    result = {k: float(v) for k, v in result.items()}
    return result


def get_max_edge_amplitude(E):
    """
    Calculate the maximum field amplitude at the edges of the grid.

    Parameters
    ----------
    E : tuple of np.array
        Electric field components as a tuple of three numpy arrays (Ex, Ey, Ez).

    Returns
    -------
    float
        The maximum field amplitude at the edges of the grid.

    Notes
    -----
    Electric field is assumed to be complex.
    """
    Ex, Ey, Ez = E
    nx, ny, nz = Ex.shape
    amplitude = np.sqrt(abs(Ex)**2 + abs(Ey)**2 + abs(Ez)**2)
    edges = [np.max(amplitude[0,:,:]), np.max(amplitude[-1,:,:]),
             np.max(amplitude[:,0,:]), np.max(amplitude[:,-1,:]),
             np.max(amplitude[:,:,0]), np.max(amplitude[:,:,-1])]
    amplitude_edge = np.max(edges)
    print(f"Max intensity at edges: {amplitude_edge/amplitude.max():.2e}")
    return amplitude_edge


def get_intensity(field, t):
    """
    Calculate the field intensity at a given time.

    Parameters
    ----------
    field : object
        Field object with a ``calculate_field`` method.
    t : float
        The time at which to calculate the field intensity.

    Returns
    -------
    np.array
        The calculated field intensity.

    Notes
    -----
    The intensity is calculated as (abs(E)**2 + abs(B)**2)/2
    """
    E, B = field.calculate_field(t=t)
    E, B = [np.real(Ex) for Ex in E], [np.real(Bx) for Bx in B]
    intensity = (E[0]**2 + E[1]**2 + E[2]**2 + B[0]**2 + B[1]**2 + B[2]**2)/2
    return intensity