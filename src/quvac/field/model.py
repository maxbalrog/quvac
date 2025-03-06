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
from scipy.spatial.transform import Rotation

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
            - 'envelope_kwargs' : dict
                Envelope specific arguments.
                For 'gauss':
                    - 'w' : float
                        Waist size.
    grid : quvac.grid.GridXYZ
        Spatial and grid.
    """
    def __init__(self, field_params, grid):
        self.grid_xyz = grid
        self.__dict__.update(self.grid_xyz.__dict__)

        angles = "theta phi".split()
        for key, val in field_params.items():
            if key in angles:
                val *= pi / 180.0
            setattr(self, key, val)

        self.beta = getattr(self, "beta", 0.)
        self.field_inhom = getattr(self, "field_inhom", "magnetic")
        # rotate coordinate grid
        self.rotate_coordinates()

        # get envelope
        self.get_envelope()

        self.E_expr = f"E0 * {self.envelope}"

        EB_keys = "Ex Ey Ez Bx By Bz".split()
        for key in EB_keys:
            setattr(self, key, 0.)

        if "W" in field_params:
            self.E0 = 1.
            self.check_energy()

        # define field calculation dict
        # self.field_dict = {
        #     "E0": self.E0,
        #     "x": self.x,
        #     "y": self.y,
        #     "z": self.z,
        # }
        # self.field_dict.update(self.envelope_kwargs)

    def get_rotation(self):
        """
        Defines the rotation transforming (0,0,1) -> (kx,ky,kz) 
        for vectors and (1,0,0) -> e(beta) = e1*cos(beta) + e2*sin(beta).
        """
        self.rotation = Rotation.from_euler("ZYZ", (self.phi, self.theta, self.beta))
        self.rotation_m = self.rotation.as_matrix()
        # Inverse rotation: (kx,ky,kz) -> (0,0,1)
        self.rotation_bwd = self.rotation.inv()
        self.rotation_bwd_m = self.rotation_bwd.as_matrix()

    def rotate_coordinates(self):
        """
        Rotates the coordinate grid.
        """
        self.get_rotation()
        self.x, self.y, self.z = self.xyz
        # axes = "xyz"
        # x_, y_, z_ = self.xyz
        # for i, ax in enumerate(axes):
        #     mx, my, mz = self.rotation_bwd_m[i, :]
        #     self.__dict__[ax] = ne.evaluate(
        #         "mx*x_ + my*y_ + mz*z_", global_dict=self.__dict__
        #     )
    
    def get_envelope(self):
        match self.envelope_type:
            case "gauss":
                self.envelope = "exp(-(2*z/w)**2)"
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
        
        if E_out is None:
            E_out = [np.zeros(self.grid_shape, dtype=config.FDTYPE) for _ in range(3)]
        if B_out is None:
            B_out = [np.zeros(self.grid_shape, dtype=config.FDTYPE) for _ in range(3)]

        # Transform to the original coordinate frame
        for i, (Ei, Bi) in enumerate(zip(E_out, B_out)):
            mx, my, mz = self.rotation_m[i, :]
            Ei += ne.evaluate("mx*Ex + my*Ey + mz*Ez", global_dict=self.__dict__).astype(config.FDTYPE)
            Bi += ne.evaluate("mx*Bx + my*By + mz*Bz", global_dict=self.__dict__).astype(config.FDTYPE)

        return E_out, B_out
