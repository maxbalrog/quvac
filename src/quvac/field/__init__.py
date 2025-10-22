"""
Various field profiles and interfaces to unite them.

.. note::
    After implementing a new field profile, it should be added to the 
    ``ANALYTIC_FIELDS`` and ``SPATIAL_MODEL_FIELDS`` dicts in the 
    ``quvac.field.__init__``. These dictionaries connect ``field_type``
    keywords in the ``ini.yaml`` files and particular field implementations.
"""
from quvac.field.dipole import DipoleAnalytic
from quvac.field.gaussian import GaussianAnalytic, GaussianSpectral
from quvac.field.laguerre_gaussian import LaguerreGaussianAnalytic
from quvac.field.model import EBInhomogeneity
from quvac.field.planewave import PlaneWave

ANALYTIC_FIELDS = {
    "dipole_analytic": DipoleAnalytic,
    "paraxial_gaussian_analytic": GaussianAnalytic,
    "laguerre_gaussian_analytic": LaguerreGaussianAnalytic,
    "eb_inhomogeneity": EBInhomogeneity,
    "plane_wave": PlaneWave,
}

SPATIAL_MODEL_FIELDS = {
    "dipole_maxwell": DipoleAnalytic,
    "paraxial_gaussian_maxwell": GaussianAnalytic,
    "paraxial_gaussian_spectral_maxwell": GaussianSpectral,
    "laguerre_gaussian_maxwell": LaguerreGaussianAnalytic,
}