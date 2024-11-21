'''
This script provides uniform ExternalField class to unite all
participating fields in one interface
'''
import os
import logging

from quvac.field.abc import Field
from quvac.field.gaussian import GaussianAnalytic
from quvac.field.maxwell import MaxwellMultiple


logger = logging.getLogger('simulation')


class ExternalField(Field):
    '''
    Class to unite several participating fields under
    one interface

    Parameters:
    -----------
    fields_params: list of dicts (field_params)
        External fields
    grid: (1d-np.array, 1d-np.array, 1d-np.array)
        xyz spatial grid to calculate fields on 
    '''
    def __init__(self, fields_params, grid, nthreads=None):
        self.fields = []
        self.grid_xyz = grid

        self.nthreads = nthreads if nthreads else os.cpu_count()

        maxwell_params = [params for params in fields_params
                          if params['field_type'].endswith('maxwell')]
        new_params = [params for params in fields_params
                      if not params['field_type'].endswith('maxwell')]
        if maxwell_params:
            new_params.append(maxwell_params)

        logger.info(f'{self.__class__.__name__}\n'
                    '----------------------------------------------------')
        for field_params in new_params:
            self.setup_field(field_params)
        logger.info('----------------------------------------------------')

    def setup_field(self, field_params):
        if isinstance(field_params, list):
            field_type = 'maxwell'
        else:
            field_type = field_params["field_type"]

        match field_type:
            case "paraxial_gaussian_analytic":
                field = GaussianAnalytic(field_params, self.grid_xyz)
            case "maxwell":
                field = MaxwellMultiple(field_params, self.grid_xyz, nthreads=self.nthreads)
            case _:
                raise NotImplementedError(f"We do not support '{field_type}' field type")
        self.fields.append(field)
        logger.info(f'Base class: {field.__class__.__name__}')
            
    def calculate_field(self, t, E_out=None, B_out=None):
        for field in self.fields:
            E_out, B_out = field.calculate_field(t, E_out=E_out, B_out=B_out)
        return E_out, B_out
    

class ProbePumpField(Field):
    '''
    Class for splitting fields into probe and pump

    Parameters:
    -----------
    fields_params: list of dicts (field_params)
        External fields
    grid: (1d-np.array, 1d-np.array, 1d-np.array)
        xyz spatial grid to calculate fields on
    probe_pump_idx: dict
        Required keys: probe, pump
        Specifies which fields are pump and probe
    '''
    def __init__(self, fields_params, grid, probe_pump_idx=None, nthreads=None):
        if not probe_pump_idx:
            probe_pump_idx = {
                'probe': [0],
                'pump': [1]
            }
        self.probe_pump_idx = probe_pump_idx

        self.nthreads = nthreads if nthreads else os.cpu_count()

        probe_params = [fields_params[idx] for idx in probe_pump_idx['probe']]
        pump_params = [fields_params[idx] for idx in probe_pump_idx['pump']]

        self.probe_field = ExternalField(probe_params, grid, nthreads=nthreads)
        self.pump_field = ExternalField(pump_params, grid, nthreads=nthreads)
            
    def calculate_field(self, t, E_probe=None, B_probe=None,
                        E_pump=None, B_pump=None):
        self.probe_field.calculate_field(t, E_out=E_probe, B_out=B_probe)
        self.pump_field.calculate_field(t, E_out=E_pump, B_out=B_pump)
        return (E_probe, B_probe), (E_pump, B_pump)

