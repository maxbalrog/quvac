'''
This script provides uniform ExternalField class to unite all
participating fields in one interface
'''

from quvac.field.abc import Field
from quvac.field.paraxial_gaussian import ParaxialGaussianAnalytic


class ExternalField(Field):
    '''
    Class to unite several participating fields under
    one interface
    '''
    def __init__(self, fields_params, grid):
        self.fields = []
        self.grid = [ax.flatten() for ax in grid]
        self.grid_shape = [dim.size for dim in grid]

        for field_params in fields_params:
            self.setup_field(field_params)

    def setup_field(self, field_params):
        field_type = field_params["field_type"]
        match field_type:
            case "paraxial_gaussian_analytic":
                field = ParaxialGaussianAnalytic(field_params, self.grid)
            case _:
                raise NotImplementedError(f"We do not support '{field_type}' field type")
        self.fields.append(field)
            
    def calculate_field(self, t, E_out=None, B_out=None):
        for field in self.fields:
            field.calculate_field(t, E_out=E_out, B_out=B_out)

