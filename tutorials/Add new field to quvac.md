# How to add new field profile to quvac

Here it is assumed that you cloned `quvac` from Github repository and have access to source sripts.

Field classes are located in `quvac.field`. In this tutorial we will implement a new field class `ConstantE`.

**Step 1. Add new field class to a separate script.**

Create a script `constant_e.py` with the following content.

```python
from quvac.field.abc import Field


class ConstantE(Field):
    '''
    Constant electric field along chosen direction.

    E = E0 * nE, nE - fixed vector

    Our field has 3 parameters:
        - E0 : float
            Field amplitude.
        - theta, phi : float, optional
            Spherical angles (in degrees) for the orientation of
            electric field, default is 0.
    '''
    def __init__(self, field_params, grid):
        super().__init__(field_params, grid)
        self.theta = getattr(self, "theta", 0)
        self.phi = getattr(self, "phi", 0)
        
        # create rotation matrices
        self.rotate_coordinates(rotate_grid=False)

    def calculate_field(self, t, E_out=None, B_out=None):
        # define our field in fixed coordinate frame
        self.Ez = self.E0
        self.Ex = self.Ey = self.Bx = self.By = self.Bz = 0.

        # rotate back to the simulation coordinate frame
        E_out, B_out = self.rotate_fields_back(E_out, B_out, mode="real")
        return E_out, B_out
```

The base class `quvac.field.abc.Field` has some basic useful methods which you could use.

**Step 2. Choose keyword for the new class.**

Since `quvac.simulation` script works with initialization files, we need to set a new keyword for the new class. All keywords are stored in two dictionaries:

1. `quvac.field.ANALYTIC_FIELDS` stores analytic profiles that could be called directly with `cls.calculate_field()` method.
2. `quvac.field.SPATIAL_MODEL_FIELDS` stores profiles that could be used as initialization to linear Maxwell propagator.

Add new entries to these dictionaries with the chosen keyword (it would be `constant_e` in our case). Now your `field.__init__.py` file should look similar to this:

```python
from quvac.field.dipole import DipoleAnalytic
from quvac.field.gaussian import GaussianAnalytic
from quvac.field.model import EBInhomogeneity
from quvac.field.constant_e import ConstantE # added

ANALYTIC_FIELDS = {
    "dipole_analytic": DipoleAnalytic,
    "paraxial_gaussian_analytic": GaussianAnalytic,
    "eb_inhomogeneity": EBInhomogeneity,
    "constant_e": ConstantE, # added
}

SPATIAL_MODEL_FIELDS = {
    "dipole_maxwell": DipoleAnalytic,
    "paraxial_gaussian_maxwell": GaussianAnalytic,
}
```

**Step 3. Run a simulation.**

Now, add an appropriate `field_params` dictionary to the `ini.yaml` file, e.g.

```python
field_params = {
    "field_type": "constant_e",
    "E0": 1,
    "theta": 90,
    "phi": 0,
}
```

Launch a simulation.
