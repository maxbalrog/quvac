"""
Analytic expression for paraxial gaussian (0-order and higher-orders).

----

.. [1] Y. I. Salamin. "Fields of a Gaussian beam beyond the paraxial 
    approximation." Applied Physics B 86 (2007): 319-326.
.. [2] A. Blinne, et al. "All-optical signatures of quantum vacuum nonlinearities in 
    generic laser fields." PRD 99.1 (2019): 016006 `(article) <https://arxiv.org/abs/1811.08895>`_.
"""

import numexpr as ne
import numpy as np
from scipy.constants import c, pi

from quvac.field.abc import ExplicitField, SpectralField
from quvac.grid import get_ek, get_polarization_vector


class GaussianAnalytic(ExplicitField):
    """
    Analytic expression for paraxial Gaussian beam.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters. Required keys are:
            - 'focus_x' : tuple of float
                Location of spatial focus (x, y, z).
            - 'focus_t' : float
                Location of temporal focus.
            - 'theta' : float
                Polar angle of k-vector (in degrees).
            - 'phi' : float
                Azimuthal angle of k-vector (in degrees).
            - 'beta' : float
                Polarization angle (in degrees).
            - 'lam' : float
                Wavelength of the pulse.
            - 'w0' : float
                Waist size.
            - 'tau' : float
                Duration.
            - 'phase0' : float
                Phase delay at focus.
            - 'E0' : float, optional
                Amplitude (either E0 or W is required).
            - 'W' : float, optional
                Energy (either E0 or W is required).
            - 'alpha_chirp' : float, optional
                Linear chirp in time domain.

    grid : quvac.grid.GridXYZ
        Spatial and grid.

    Notes
    -----
    All field parameters are in SI units.

    Higher-order paraxial Gaussian orders are from [1]_.

    For circular polarization we abide by the following convention:
        - `right-circular` corresponds to (ex + i*ey)
        - `left-circular` to (ex - i*ey)
        
    Difference in field amplitudes for linear and circular polarization should 
    be automatically taken care of by energy correction.
    """

    def __init__(self, field_params, grid):
        super().__init__(field_params, grid)

        self.order = getattr(self, "order", 0)
        self.polarization = getattr(self, "polarization", "linear")

        if "E0" not in field_params:
            err_msg = ("Field params need to have either W (energy) or"
                       "E0 (amplitude) as key")
            assert "W" in field_params, err_msg
            self.E0 = 1.0e10

        # Define additional field variables
        self.x0, self.y0, self.z0 = self.focus_x
        self.t0 = self.focus_t
        self.B0 = self.E0 / c
        self.k = 2.0 * pi / self.lam
        self.omega = c * self.k
        self.zR = pi * self.w0**2 / self.lam

        # Rotate coordinate grid
        self.rotate_coordinates()

        # Define variables not depending on time step
        self.w = "(w0 * sqrt(1. + (z/zR)**2))"
        self.r2 = "(x**2 + y**2)"
        # self.R = "(z + zR**2/z)"
        self.R_inv = "z/(z**2 + zR**2)"
        self.E_expr = f"B0 * w0/{self.w} * exp(-{self.r2}/{self.w}**2)"
        self.phase_no_t = ne.evaluate(
            f"phase0 - k*{self.r2}*{self.R_inv}/2. + arctan(z/zR)",
            global_dict=self.__dict__,
        )
        # self.phase_no_t = ne.evaluate(
        #     f"phase0 - k*{self.r2}/(2.*{self.R}) + arctan(z/zR)",
        #     global_dict=self.__dict__,
        # )

        self.E = ne.evaluate(self.E_expr, global_dict=self.__dict__)

        if self.order > 0:
            self.calculate_ho_orders()

        # Set up correct field amplitude
        if "W" in field_params:
            self.check_energy()

    def define_ho_variables(self):
        """
        Define higher order variables.
        """
        self.eps = self.w0 / self.zR
        self.xi = ne.evaluate("x/w0", global_dict=self.__dict__)
        self.nu = ne.evaluate("y/w0", global_dict=self.__dict__)
        self.zeta = ne.evaluate("z/zR", global_dict=self.__dict__)
        self.rho = ne.evaluate("sqrt(xi**2 + nu**2)", global_dict=self.__dict__)
        self.f = ne.evaluate(
            "exp(-1j*arctan(zeta))/sqrt(1 + zeta**2)", global_dict=self.__dict__
        )
        for n in range(1, 5):
            self.__dict__[f"f{n}"] = ne.evaluate(
                "1/(1+zeta**2)**(n/2) * exp(-1j*n*arctan(zeta))",
                global_dict=self.__dict__,
            )

        self.Ex_terms = {
            0: "1",
            2: "eps**2 * f2 * (xi**2 - f1*rho**4/4)",
            4: (
                "eps**4 * f2 * (1/8 - f1*rho**2/4 + f2*rho**2*(xi**2 - rho**2/16) - "
                "f3*rho**4*(xi**2/4 + rho**2/8) + f4*rho**8/32)"
            ),
        }
        self.Ey_terms = {
            0: "0",
            2: "eps**2 * f2",
            4: "eps**4*f4*rho**2 * (1 - f1*rho**2/4)",
        }
        self.Ez_terms = {
            0: "0",
            1: "eps * f1",
            3: "eps**3 * f2 * (-0.5 + f1*rho**2 - f2*rho**4/4)",
            5: (
                "eps**5 * f3 * (-3/8 - 3*f1*rho**2/8 + 17*f2*rho**4/16 - "
                "3*f3*rho**6/8 + f4*rho**8/32)"
            ),
        }
        self.By_terms = {
            0: "1",
            2: "eps**2 * f2*rho**2 * (1/2 - f1*rho**2/4)",
            4: ("eps**4 * f2 * (-1/8 + f1*rho**2/4 + 5*f2*rho**4/16 - f3*rho**6/4 + "
                "f4*rho**8/32)"),
        }
        self.Bz_terms = {
            0: "0",
            1: "eps * f1",
            3: "eps**3*f2 * (1/2 + f1*rho**2/2 - f2*rho**4/4)",
            5: ("eps**5*f3 * (3/8 + 3*f1*rho**2/8 + 3*f2*rho**4/16 - f3*rho**6/4 + "
                "f4*rho**8/32)"),
        }

    def calculate_ho_orders(self):
        """
        Calculate higher order terms for the fields.
        """
        self.define_ho_variables()
        # For a given order, combine final expression to calculate
        names = "Ex Ey Ez By Bz".split()
        for name in names:
            terms = self.__dict__[f"{name}_terms"]
            self.__dict__[f"{name}_expr"] = " + ".join(
                [term for order, term in terms.items() if order <= self.order]
            )
            self.__dict__[f"{name}_ho"] = ne.evaluate(
                self.__dict__[f"{name}_expr"], global_dict=self.__dict__
            )

    def check_energy(self):
        """
        Check and adjust the field energy.
        """
        self._check_energy()

        if self.modify_energy:
            self.E0 *= self.W_correction
            self.B0 = self.E0 / c
            self.E = ne.evaluate(self.E_expr, global_dict=self.__dict__)

    def figure_out_field_components(self, Et):
        if self.polarization == "linear":
            if self.order > 0:
                self.Ex = ne.evaluate("Et * Ex_ho", global_dict=self.__dict__)
                self.Ey = ne.evaluate("Et * Ey_ho * xi * nu",
                                      global_dict=self.__dict__)
                self.Ez = ne.evaluate("-1j*Et * Ez_ho * xi", global_dict=self.__dict__)
                self.Bx = 0.0
                self.By = ne.evaluate("Et * By_ho", global_dict=self.__dict__)
                self.Bz = ne.evaluate("-1j*Et * Bz_ho * nu", global_dict=self.__dict__)
            else:
                self.Ex = self.By = Et.copy()
                self.Ey = self.Ez = self.Bx = self.Bz = 0.0
        elif self.polarization in ["left-circular", "right-circular"]:
            if self.order > 0:
                raise NotImplementedError("Higher paraxial orders for circular "
                                          "polarization are not supported")
            else:
                self.Ex = self.By = Et.copy()
                if self.polarization == "right-circular":
                    self.Ey = 1j * Et.copy()
                else:
                    self.Ey = -1j * Et.copy()
                self.Bx = -self.Ey
                self.Ez = self.Bz = 0.0
        else:
            raise NotImplementedError(f"`{self.polarization}` polarization is not "
                                      "supported")
        
    def get_plane_wave_phase(self, t):
        alpha_chirp = getattr(self, "alpha_chirp", 0)
        plane_phase = "(t-t0-z/c)"
        if alpha_chirp != 0:
            psi_plane_expr = f"({plane_phase}*(omega + alpha_chirp*{plane_phase}/tau))"
        else:
            psi_plane_expr = f"(omega*{plane_phase})"
        psi_plane = ne.evaluate(psi_plane_expr, global_dict=self.__dict__,
                                local_dict={'c': c, 't': t, 'omega': self.omega,
                                            'alpha_chirp': alpha_chirp})
        return psi_plane

    def calculate_field(self, t, E_out=None, B_out=None, mode="real"):
        """
        Calculates the electric and magnetic fields at a given time step.
        """
        k = 2.0 * pi / self.lam # noqa: F841
        self.psi_plane = self.get_plane_wave_phase(t)
        self.phase = "(phase_no_t + psi_plane)"

        Et = ne.evaluate(
            f"E * exp(-(2.*psi_plane/(omega*tau))**2) * exp(-1.j*{self.phase})",
            global_dict=self.__dict__,
        )

        self.figure_out_field_components(Et)

        if mode == "real":
            self.convert_fields_to_real()

        E_out, B_out = self.rotate_fields_back(E_out, B_out, mode)
        return E_out, B_out
    

class GaussianSpectral(SpectralField):
    """
    Analytic expression for paraxial Gaussian beam.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters. Required keys are:
            - 'focus_x' : tuple of float
                Location of spatial focus (x, y, z).
            - 'focus_t' : float
                Location of temporal focus.
            - 'theta' : float
                Polar angle of k-vector (in degrees).
            - 'phi' : float
                Azimuthal angle of k-vector (in degrees).
            - 'beta' : float
                Polarization angle (in degrees).
            - 'lam' : float
                Wavelength of the pulse.
            - 'w0' : float
                Waist size.
            - 'tau' : float
                Duration.
            - 'phase0' : float
                Phase delay at focus.
            - 'E0' : float, optional
                Amplitude (either E0 or W is required).
            - 'W' : float, optional
                Energy (either E0 or W is required).
            - 'alpha_chirp' : float, optional
                Linear chirp in frequency domain.

    grid : quvac.grid.GridXYZ
        Spatial and grid.

    Notes
    -----
    All field parameters are in SI units.

    The expression for the spectrum is taken from [2]_ (Eq. 24).
    """

    def __init__(self, field_params, grid):
        super().__init__(field_params, grid)

        if "E0" not in field_params:
            err_msg = ("Field params need to have either W (energy) or"
                       "E0 (amplitude) as key")
            assert "W" in field_params, err_msg
            self.E0 = 1.0e10

        # Define additional field variables
        self.x0, self.y0, self.z0 = self.focus_x
        self.t0 = self.focus_t
        self.B0 = self.E0 / c
        self.k = 2.0 * pi / self.lam
        self.omega = c * self.k
        self.alpha_chirp = getattr(self, "alpha_chirp", 0)

        # Rotate k-space grid
        self.rotate_kgrid()

        self.kperp2 = "(kx**2 + ky**2)"
        self.define_vector_potential_expression()
        self.vector_potential = ne.evaluate(self.vector_potential_expr,
                                            local_dict=self.vector_potential_dict)
        # self.vector_potential = np.fft.ifftshift(self.vector_potential)
        
        A = self.rotate_vector_potential_back()
        self.Ax, self.Ay, self.Az = [np.fft.ifftshift(Ai) for Ai in A]
        # self.Ax, self.Ay, self.Az = self.rotate_vector_potential_back()

    def define_vector_potential_expression(self):
        self.vector_potential_expr = (
            "where(kz > 0, pi**1.5/2 * 1/(1j*kabs) * kz/kabs * E0*tau*w0**2 * "
            f"exp(-(w0/2)**2*{self.kperp2}) * "
            "exp(-(tau/4)**2*(c*kabs-omega)**2*(1-1j*alpha)), 0)"
        )
        self.vector_potential_dict = {
            "pi": pi,
            "c": c,
            "kx": self.kx_rotated,
            "ky": self.ky_rotated,
            "kz": self.kz_rotated,
            "kabs": self.kabs_rotated,
            "E0": self.E0,
            "tau": self.tau,
            "w0": self.w0,
            "omega": self.omega,
            "alpha": self.alpha_chirp,
        }
    
    def calculate_field(self, t, E_out=None, B_out=None, mode="real"):
        """
        Calculates the electric and magnetic fields at a given time step.
        """
        raise NotImplementedError("GaussianSpectral works only as a model field" \
        "for MaxwellField")
    

class GaussianSpectralDirect(SpectralField):
    """
    Analytic expression for paraxial Gaussian beam.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters. Required keys are:
            - 'focus_x' : tuple of float
                Location of spatial focus (x, y, z).
            - 'focus_t' : float
                Location of temporal focus.
            - 'theta' : float
                Polar angle of k-vector (in degrees).
            - 'phi' : float
                Azimuthal angle of k-vector (in degrees).
            - 'beta' : float
                Polarization angle (in degrees).
            - 'lam' : float
                Wavelength of the pulse.
            - 'w0' : float
                Waist size.
            - 'tau' : float
                Duration.
            - 'phase0' : float
                Phase delay at focus.
            - 'E0' : float, optional
                Amplitude (either E0 or W is required).
            - 'W' : float, optional
                Energy (either E0 or W is required).
            - 'alpha_chirp' : float, optional
                Linear chirp in frequency domain.

    grid : quvac.grid.GridXYZ
        Spatial and grid.

    Notes
    -----
    All field parameters are in SI units.

    The expression for the spectrum is taken from [2]_ (Eq. 24).
    """

    def __init__(self, field_params, grid):
        super().__init__(field_params, grid)

        if "E0" not in field_params:
            err_msg = ("Field params need to have either W (energy) or"
                       "E0 (amplitude) as key")
            assert "W" in field_params, err_msg
            self.E0 = 1.0e10

        # Define additional field variables
        self.x0, self.y0, self.z0 = self.focus_x
        self.t0 = self.focus_t
        self.B0 = self.E0 / c
        self.k = 2.0 * pi / self.lam
        self.omega = c * self.k
        self.alpha_chirp = getattr(self, "alpha_chirp", 0)

        self.get_vector_potential()

    def get_vector_potential(self):
        # kx, ky, kz = [np.fft.fftshift(k) for k in self.kmeshgrid]
        kx, ky, kz = self.kmeshgrid
        k0x, k0y, k0z = get_ek(self.theta, self.phi)
        klong = k0x*kx + k0y*ky + k0z*kz
        kperpx, kperpy, kperpz = (kx - klong*k0x, ky - klong*k0y, kz - klong*k0z)
        kperp2 = kperpx**2 + kperpy**2 + kperpz**2
        kabs = np.sqrt(kx**2 + ky**2 + kz**2)

        self.vector_potential_expr = (
            "where(klong > 0, pi**1.5/2 * 1/(1j*kabs) * klong/kabs * E0*tau*w0**2 * "
            "exp(-(w0/2)**2*kperp2) * "
            "exp(-(tau/4)**2*(c*kabs-omega)**2*(1-1j*alpha)), 0)"
        )

        self.vector_potential_dict = {
            "pi": pi,
            "c": c,
            "kx": kx,
            "ky": ky,
            "kz": kz,
            "kabs": kabs,
            "klong": klong,
            "kperp2": kperp2,
            "E0": self.E0,
            "tau": self.tau,
            "w0": self.w0,
            "omega": self.omega,
            "alpha": self.alpha_chirp,
        }

        x0, y0, z0 = [ax[0] for ax in self.grid]
        dft_factor = np.exp(1j*(kx*x0 + ky*y0 +kz*z0))

        vector_potential = ne.evaluate(self.vector_potential_expr,
                                       local_dict=self.vector_potential_dict)
        vector_potential *= dft_factor
        
        ebeta = get_polarization_vector(self.theta, self.phi, self.beta)
        self.Ax, self.Ay, self.Az = [ei*vector_potential for ei in ebeta]
        # self.Ax, self.Ay, self.Az = [np.fft.ifftshift(ei*vector_potential) for ei in ebeta]
    
    def calculate_field(self, t, E_out=None, B_out=None, mode="real"):
        """
        Calculates the electric and magnetic fields at a given time step.
        """
        raise NotImplementedError("GaussianSpectral works only as a model field" \
        "for MaxwellField")
