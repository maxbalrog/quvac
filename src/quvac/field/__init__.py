"""
Various field profiles and interfaces to unite them.

.. note::
    After implementing a new field profile, it should be added to the 
    ``ANALYTIC_FIELDS`` and ``SPATIAL_MODEL_FIELDS`` dicts in the 
    ``quvac.field.__init__``. These dictionaries connect ``field_type``
    keywords in the ``ini.yml`` files and particular field implementations.
"""
from quvac.field.dipole import DipoleAnalytic
from quvac.field.gaussian import (
    GaussianAnalytic,
    GaussianEllipticAnalytic,
    GaussianSpectral,
)
from quvac.field.laguerre_gaussian import LaguerreGaussianAnalytic
from quvac.field.model import EBInhomogeneity
from quvac.field.planewave import PlaneWave

ANALYTIC_FIELDS = {
    "dipole_analytic": DipoleAnalytic,
    "paraxial_gaussian_analytic": GaussianAnalytic,
    "elliptic_gaussian_analytic": GaussianEllipticAnalytic,
    "laguerre_gaussian_analytic": LaguerreGaussianAnalytic,
    "eb_inhomogeneity": EBInhomogeneity,
    "plane_wave": PlaneWave,
}

SPATIAL_MODEL_FIELDS = {
    "dipole_maxwell": DipoleAnalytic,
    "paraxial_gaussian_maxwell": GaussianAnalytic,
    "elliptic_gaussian_maxwell": GaussianEllipticAnalytic,
    "paraxial_gaussian_spectral_maxwell": GaussianSpectral,
    "laguerre_gaussian_maxwell": LaguerreGaussianAnalytic,
}