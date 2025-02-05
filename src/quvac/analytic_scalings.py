"""
Here we provide utility functions to calculate analytic 
scalings from articles
"""

import numpy as np
from scipy.constants import alpha, c, hbar, m_e, pi
from scipy.integrate import quad
from scipy.special import erfc

W_e = m_e * c**2  # electron's rest energy
lam_C = hbar / (m_e * c)  # reduced Compton wavelength


def get_two_paraxial_scaling(fields, channels="both"):
    """
    This is a variation of Eq.(25) from F. Karbstein, et al. "Vacuum
    birefringence at x-ray free-electron lasers." New Journal of Physics
    23.9 (2021): 095001.

    We introduced additional polarization-dependent factors (beta). The
    formula should work for theta not close to 180, loose focusing and short
    pulse duration.

    Note: result for N_perp should be used only for beta=45
    """
    theta_c = (fields[1]["theta"] - fields[0]["theta"]) * pi / 180
    beta = (fields[1]["beta"] - fields[0]["beta"]) * pi / 180

    omega = 2 * pi * c / fields[0]["lam"]
    N_signal, N_perp = 0, 0
    # Iterate through pump/probe channels
    match channels:
        case "first":
            fields_to_iterate = [fields]
        case "second":
            fields_to_iterate = [fields[::-1]]
        case _:
            fields_to_iterate = [fields, fields[::-1]]

    for field1, field2 in fields_to_iterate:
        tau1, tau2 = field1["tau"], field2["tau"]
        w01, w02 = field1["w0"], field2["w0"]
        W1, W2 = field1["W"], field2["W"]

        # Define additional parameters
        T1 = tau1 * np.sqrt(1 + 2 * (tau1 / tau2) ** 2)
        T2 = tau2 * np.sqrt(1 + 0.5 * (tau2 / tau1) ** 2)
        w1 = w01 * np.sqrt(1 + 2 * (w01 / w02) ** 2)
        w2 = w02 * np.sqrt(1 + 0.5 * (w02 / w01) ** 2)
        sin_term = T1 * T2 * c**2 / (w1 * w2) * np.sin(theta_c) ** 2
        cos_term = 4 * (1 - np.cos(theta_c)) ** 2

        # Calculate N_signal and N_perp
        theta_term = (1 - np.cos(theta_c)) ** 4 / np.sqrt(cos_term + sin_term)
        theta_term *= 1 / np.sqrt((T1 / tau1 * w01 / w1) ** 2 * cos_term + sin_term)
        dims = hbar * omega / W_e
        prefactor = 8 * pi**2 / 225 * (alpha / pi) ** 4 * W1 * W2**2 / W_e**3 * dims
        prefactor *= T1 / tau1 * w1 / w01 * lam_C**4 / (w1 * w2) ** 2 / 9
        N_perp += prefactor * theta_term * 9 * np.sin(2 * beta) ** 2
        N_signal += prefactor * theta_term * (130 - 66 * np.cos(2 * beta))
    return N_signal, N_perp


def f_felix(x, r, k0=20):
    def integrand(k):
        s = 0
        for l in [-1, 1]:
            s += np.exp(2*l*r*x*k) * erfc(l*r*k + x)
        s = np.abs(s)**2
        return np.exp(-k**2) * s
        
    prefactor = np.sqrt((1 + 2*r**2)/3) * x**2 * np.exp(2*x**2)
    result = quad(integrand, -k0, k0)
    return prefactor * result[0]


def get_onaxis_scaling(fields, k0=20):
    """
    This is an integrated version of Eq.(11) [Eq. (13)] from E. Mosman, F.Karbstein
    "Vacuum birefringence and diffraction at XFEL: from analytical estimates to optimal parameters"
    PRD 104.1 (2021): 013006.
    
    We assume that 1st field in the list is x-ray

    Parameters:
    -----------
    k0: float
        Integration limit for the integral in F function
    """
    wx, w0 = fields[0]["w0"], fields[1]["w0"]
    beta = wx / w0
    Wx, W = fields[0]["W"], fields[1]["W"]
    lam_x, lam = fields[0]["lam"], fields[1]["lam"]
    omega_x = 2 * pi * c / lam_x
    Nx = Wx / (hbar * omega_x)
    T, tau = fields[0]["tau"], fields[1]["tau"]
    zR = pi * w0**2 / lam

    x0 = 4 * zR / (c * np.sqrt(T**2 + 1/2*tau**2))
    r0 = T / tau
    Fbeta = f_felix(x0 * np.sqrt(1+2*beta**2), r0, k0=k0)
    F0 = f_felix(x0, r0, k0=k0)

    prefactor = 4 * alpha**4 / (3*pi)**1.5 * W**2 * (hbar * omega_x)**2 / W_e**4
    result = prefactor * (lam_C / w0)**4 / (1 + 2*beta**2) * np.sqrt(Fbeta*F0) * Nx

    N_signal = 26/45 * result
    N_perp = 1/25 * result
    return N_signal, N_perp
