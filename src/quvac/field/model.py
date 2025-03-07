"""
Model fields.

Currently implements:

1. Model electric/magnetic field with inhomogeneity in one direction
and infinite transverse extend in others. Adapted from [1]_.

----

.. [1] H. Gies, F. Karbstein, and N. Seegert. "Quantum reflection
   as a new signature of quantum vacuum nonlinearity." NJP 15.8 (2013): 
   083002.
"""

import numexpr as ne
import numpy as np
from scipy.constants import c, pi

from quvac.field.abc import Field
from quvac.field.utils import get_field_energy
from quvac import config


class EBInhomogeneity(Field):
    """
    Model inhomogeneities of electric/magnetic field.

    Field is of type B = B(z) * eB.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters. Required keys are:
            - 'field_inhom': str
                Field inhomogeneity, either 'electric' or 'magnetic'.
            - 'theta' : float
                Polar angle of eB-vector (in degrees).
            - 'phi' : float
                Azimuthal angle of eB-vector (in degrees).
            - 'E0' : float
                Amplitude.
            - 'envelope_type' : str
                Envelope type, by default 'gauss'.
            Envelope specific arguments:
                For 'gauss':
                    - 'w0' : float
                        Waist size.
                For 'gauss-modulated':
                    - 'w0' : float
                        Waist size.
                    - 'lam': float
                        Wavelength of modulation.
                    - 'phase0': float, optional
                        Initial phase
    grid : quvac.grid.GridXYZ
        Spatial and grid.
    """
    def __init__(self, field_params, grid):
        super().__init__(field_params, grid)

        self.beta = getattr(self, "beta", 0.)
        self.field_inhom = getattr(self, "field_inhom", "magnetic")
        
        # create rotation matrices to transform the field orientation
        # without changing the grid
        self.rotate_coordinates(rotate_grid=False)

        self.get_envelope()
        self.E_expr = f"E0 * {self.envelope}"

        EB_keys = "Ex Ey Ez Bx By Bz".split()
        for key in EB_keys:
            setattr(self, key, 0.)

        if "W" in field_params:
            self.E0 = 1.
            self.check_energy()
    
    def get_envelope(self):
        """
        Define the envelope function for the field.

        The envelope function is defined based on the `envelope_type` parameter. 
        Currently, only the Gaussian envelope type is supported.

        Raises
        ------
        NotImplementedError
            If the `envelope_type` is not supported.
        """
        match self.envelope_type:
            case "gauss":
                self.envelope = "exp(-(z/w0)**2)"
            case "gauss-modulated":
                self.phase0 = getattr(self, "phase0", 0.)
                self.kz_m = 2*pi/self.lam
                self.envelope = "exp(-(z/w0)**2) * cos(kz_m*z + phase0)"
            case _:
                raise NotImplementedError(f"`{self.envelope_type}` envelope type"
                                          "is not supported")
            
    def check_energy(self):
        """
        Check and adjust the field energy.
        """
        E, B = self.calculate_field(t=0)
        W = get_field_energy(E, B, self.dV)

        if "W" in self.__dict__.keys() and not np.isclose(W, self.W, rtol=1e-5):
            self.E0 *= np.sqrt(self.W / W)
            self.W_num = W * self.E0**2
    
    def calculate_field(self, t, E_out=None, B_out=None):
        E = ne.evaluate(self.E_expr, global_dict=self.__dict__).astype(config.FDTYPE)

        if self.field_inhom == "electric":
            self.Ez = E
        elif self.field_inhom == "magnetic":
            self.Bz = E
        else:
            raise NotImplementedError(f"`{self.field_inhom}` field inhomogeneity"
                                      "is not supported")
        
        E_out, B_out = self.rotate_fields_back(E_out, B_out, mode="real")
        return E_out, B_out
