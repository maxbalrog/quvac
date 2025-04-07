"""
Analytic expression for dipole wave and function to create multibeam
configuration by compining several focused fields.

----

.. [1] I. Gonoskov et al. "Dipole pulse theory: Maximizing the field
    amplitude from 4 π focused laser pulses." PRA 86.5 (2012): 053836.

.. [2] S. S. Bulanov et al. "Multiple Colliding Electromagnetic Pulses: 
    A Way to Lower the Threshold of e+ e-Pair Production from Vacuum." 
    PRL 104.22 (2010): 220404.
"""
# ruff: noqa: F841
from copy import deepcopy

import numexpr as ne
from scipy.constants import c, pi

from quvac.field.abc import ExplicitField


class DipoleAnalytic(ExplicitField):
    """
    Analytic expression for dipole wave.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters.
            - focus_x : tuple of float
                Location of spatial focus (x, y, z).
            - focus_t : float
                Location of temporal focus.
            - theta : float
                Spherical angle of dipole moment d0 with z-axis (in degrees).
            - phi : float
                Spherical angle of dipole moment d0 with x-axis (in degrees).
            - lam : float
                Pulse wavelength.
            - tau : float
                Duration.
            - W : float, optional
                Energy.
            - envelope : str
                Envelope type, either "plane" or "gauss".
            - dipole_type : str
                Type of dipole, either "electric" or "magnetic".
    grid : quvac.grid.GridXYZ
        Spatial and spectral grid.
    
    Notes
    -----
    Dipole wave expression is from [1]_.

    d0 is along ez by default.
    """
    def __init__(self, field_params, grid):
        super().__init__(field_params, grid)
        
        self.beta = getattr(self, "beta", 0)
        self.envelope = getattr(self, "envelope", "plane")
        self.dipole_type = getattr(self, "dipole_type", "electric")

        # Define additional field variables
        self.x0, self.y0, self.z0 = self.focus_x
        self.t0 = self.focus_t
        self.k = 2.0 * pi / self.lam
        self.omega = c * self.k
        self.d0 = 1.

        # Rotate coordinate grid
        self.rotate_coordinates()

        # Define variables independent of time
        self.R_expr = 'sqrt(x**2 + y**2 + z**2)' # radius
        self.R =  ne.evaluate(self.R_expr, global_dict=self.__dict__)

        R_inv_expr = "where(R==0, 0, 1/R)"
        R_inv =  ne.evaluate(R_inv_expr, global_dict=self.__dict__)
        
        self.nx, self.ny, self.nz = [ax*R_inv for ax in (self.x, self.y, self.z)]

        self.EB_dict = {"R": self.R,
                        "c": c,
                        "k": self.k,
                        "omega": self.omega,
                        "d0": self.d0,
                        "nx": self.nx,
                        "ny": self.ny,
                        "nz": self.nz}

        self.set_envelope_expressions()
        self.check_energy()

    def set_envelope_expressions(self):
        """
        Defines envelope expressions.
        """
        if self.envelope == "plane":
            self.g_expr = "1j*exp(-1j*omega*t)"
            self.gdot_expr = 'omega*exp(-1j*omega*t)'
            self.gdotdot_expr = "-1j*omega**2*exp(-1j*omega*t)"
            self.E_R0 = "omega**3"
        elif self.envelope == "gauss":
            a2 = "(1/(tau/2)**2)"
            env = f"-1j*exp(-t**2*{a2} - 1j*omega*t)"
            self.g_expr = env
            self.gdot_expr = f'{env} * (-2*t*{a2} - 1j*omega)'
            self.gdotdot_expr = (f'{env} * (4*t**2*{a2}**2 - 2*{a2} - omega**2 + '
                                 f'4j*t*omega*{a2})')
            self.E_R0 = (f"{env} * (-2*t*{a2}*(4*t**2*{a2}**2 - 3*omega**2 - 6*{a2})"
                         f" - 1j*omega*(12*t**2*{a2}**2 - omega**2 - 6*{a2}))")
        else:
            raise NotImplementedError(f"Envelope {self.envelope} not implemented")

    def check_energy(self):
        """
        Checks and adjusts the energy of the field.
        """
        self._check_energy()

        if self.modify_energy:
            self.d0 *= self.W_correction

    def _g(self, t):
        return ne.evaluate(self.g_expr, global_dict=self.__dict__)
    
    def _gdot(self, t):
        return ne.evaluate(self.gdot_expr, global_dict=self.__dict__)
    
    def _gdotdot(self, t):
        return ne.evaluate(self.gdotdot_expr, global_dict=self.__dict__)
    
    def _g_plusminus(self, t, sign=1):
        return self._g(t-self.R/c) + sign*self._g(t+self.R/c)
    
    def _gdot_plusminus(self, t, sign=1):
        return self._gdot(t-self.R/c) + sign*self._gdot(t+self.R/c)
    
    def _gdotdot_plusminus(self, t, sign=1):
        return self._gdotdot(t-self.R/c) + sign*self._gdotdot(t+self.R/c)

    def _fix_singularity(self, t):
        # fix divergence at R=0
        Nx,Ny,Nz = self.Ex.shape
        self.Ex[Nx//2,Ny//2,Nz//2] = 0.
        self.Ey[Nx//2,Ny//2,Nz//2] = 0.
        self.Ez[Nx//2,Ny//2,Nz//2] = 4/(3*c**3)*ne.evaluate(self.E_R0,
                                                            global_dict=self.__dict__)

        self.Bx[Nx//2,Ny//2,Nz//2] = 0.
        self.By[Nx//2,Ny//2,Nz//2] = 0.
    
    def calculate_field(self, t, E_out=None, B_out=None, mode="real"):
        """
        Calculates the electric and magnetic fields at a given time step.
        """
        gdotdot_p = self._gdotdot_plusminus(t)
        gdotdot_m = self._gdotdot_plusminus(t, sign=-1)
        gdot_p = self._gdot_plusminus(t)
        gdot_m = self._gdot_plusminus(t, sign=-1)
        g_m = self._g_plusminus(t, sign=-1)

        Bt = ne.evaluate("gdotdot_p/(R*c**2) + gdot_m/(R**2*c)", 
                         global_dict=self.EB_dict)
        Et = ne.evaluate("gdot_p/(c*R**2) + g_m/R**3", global_dict=self.EB_dict)
        
        self.Ex = ne.evaluate('nx*nz*gdotdot_m/(R*c**2) + 3*nx*nz*Et',
                              global_dict=self.EB_dict)
        self.Ey = ne.evaluate('ny*nz*gdotdot_m/(R*c**2) + 3*ny*nz*Et', 
                              global_dict=self.EB_dict)
        self.Ez = ne.evaluate('-(nx**2+ny**2)*gdotdot_m/(R*c**2) + (3*nz**2-1)*Et',
                               global_dict=self.EB_dict)

        self.Bx = ne.evaluate("-ny*Bt", global_dict=self.EB_dict)
        self.By = ne.evaluate("nx*Bt", global_dict=self.EB_dict)
        self.Bz = 0.
        
        # fix divergence at R=0
        self._fix_singularity(t)

        # h-dipole (magnetic) transformation
        if self.dipole_type == "magnetic":
            self.Ex, self.Bx = -self.Bx, self.Ex
            self.Ey, self.By = -self.By, self.Ey
            self.Ez, self.Bz = -self.Bz, self.Ez

        for field in "Ex Ey Ez Bx By Bz".split():
            self.__dict__[field] *= self.d0

        if mode == "real":
            self.convert_fields_to_real()

        E_out, B_out = self.rotate_fields_back(E_out, B_out, mode)
        return E_out, B_out


def create_multibeam(params, n_beams=6, mode='belt', phi0=0, idx0=1):
    """
    Create multibeam configuration from several focused pulses to 
    approximate the dipole wave and achieve higher intensity 
    at focus.

    Parameters
    ----------
    params : dict
        Dictionary containing the field parameters.
    n_beams : int, optional
        Number of beams, by default 6.
    mode : str, optional
        Configuration mode, either 'belt' or 'sphere', by default 'belt'.
    phi0 : float, optional
        Initial angle for the beams, by default 0.
    idx0 : int, optional
        Offset index for `beams` dictionary.

    Returns
    -------
    beams : dict
        Dictionary containing the parameters for each beam.

    Notes
    -----
    Configuration follows from [2]_.
    """
    # distribute the energy
    W_per_beam = params['W'] / n_beams
    beams = {}
    if mode == "sphere":
        n_beams = n_beams // 3
        theta_arr = [0, 45, -45]
    phi_c = 360/n_beams

    for i in range(n_beams):
        params_beam = deepcopy(params)
        params_beam["W"] = W_per_beam
        params_beam["phi"] = i*phi_c + phi0
        params_beam["theta"] = 90
        match mode:
            case "belt":
                beams[f"field_{i+1+idx0}"] = params_beam
            case "sphere":
                for j,theta in enumerate(theta_arr):
                    idx = i*3 + j
                    params_phi = deepcopy(params_beam)
                    params_phi["theta"] += theta
                    beams[f"field_{idx+1+idx0}"] = params_phi
    return beams
