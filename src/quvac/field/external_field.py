"""
Common interfaces for external field and pump-probe fields to unify
various field types.
"""

import logging

from quvac.field import ANALYTIC_FIELDS
from quvac.field.abc import Field
from quvac.field.maxwell import MaxwellMultiple

_logger = logging.getLogger("simulation")


class ExternalField(Field):
    """
    Class to unite several participating external fields under one interface.

    Parameters
    ----------
    fields_params : list of dict
        List of dictionaries containing the parameters for each external field.
    grid : quvac.grid.GridXYZ
        Spatial and spectral grid.
    fft_executor: quvac.pyfftw_executor.FFTExecutor, optional
        Executor that performs FFTs.
    nthreads : int, optional
        Number of threads to use for calculations. If not provided, defaults to the 
        number of CPU cores.

    Attributes
    ----------
    fields : list
        List of field objects.
    grid_xyz : tuple of np.array
        The spatial grid.
    fft_executor: quvac.pyfftw_executor.FFTExecutor
        Executor that performs FFTs.
    nthreads : int
        Number of threads to use for calculations.
    """

    def __init__(self, fields_params, grid, fft_executor=None, nthreads=None):
        self.fields = []
        self.grid_xyz = grid

        self.nthreads = nthreads
        self.fft_executor = fft_executor

        # Reallocation of E_out and B_out fields at each time step was removed.
        # If Maxwell fields are present in the setup, the zeroing out fields is 
        # automatic
        self.need_to_zero_fields = True

        maxwell_params = [
            params
            for params in fields_params
            if params["field_type"].endswith("maxwell")
        ]
        new_params = [
            params
            for params in fields_params
            if not params["field_type"].endswith("maxwell")
        ]
        if maxwell_params:
            # MaxwellField directly overwrites given E_out and B_out arrays.
            # If it's used together with other fields (analytic profiles) then
            # it needs to go first so total external field is correct.
            new_params.insert(0, maxwell_params)
            # MaxwellField takes care of zeroing out.
            self.need_to_zero_fields = False

        _logger.info(
            f"{self.__class__.__name__}\n"
            "----------------------------------------------------"
        )
        for field_params in new_params:
            self.setup_field(field_params)
        _logger.info("----------------------------------------------------")

    def setup_field(self, field_params):
        """
        Set up a field based on the provided parameters.

        Parameters
        ----------
        field_params : dict or list of dict
            Dictionary or list of dictionaries containing the parameters for the field.

        Raises
        ------
        NotImplementedError
            If the field type is not supported.
        """
        if isinstance(field_params, list):
            field_type = "maxwell"
        else:
            field_type = field_params["field_type"]

        if field_type in ANALYTIC_FIELDS:
            field = ANALYTIC_FIELDS[field_type](field_params, self.grid_xyz)
        elif field_type == "maxwell":
            field = MaxwellMultiple(
                    field_params, self.grid_xyz, self.fft_executor, 
                    nthreads=self.nthreads
                )
        else:
            raise NotImplementedError(
                    f"We do not support '{field_type}' field type"
                )

        self.fields.append(field)
        _logger.info(f"Base class: {field.__class__.__name__}")

    def maybe_zero_out_field_buffers(self, E_out, B_out):
        output_given = (E_out is not None) and (B_out is not None)
        if output_given and self.need_to_zero_fields:
            for idx in range(3):
                E_out[idx][:] = 0.
                B_out[idx][:] = 0.
        return E_out, B_out

    def calculate_field(self, t, E_out=None, B_out=None):
        """
        Calculates the electric and magnetic fields at a given time step.
        """
        E_out, B_out = self.maybe_zero_out_field_buffers(E_out, B_out)
        for field in self.fields:
            E_out, B_out = field.calculate_field(t, E_out=E_out, B_out=B_out)
        return E_out, B_out


class ProbePumpField(Field):
    """
    Class for splitting fields into probe and pump.

    Parameters
    ----------
    fields_params : list of dict
        List of dictionaries containing the parameters for each external field.
    grid : quvac.grid.GridXYZ
        Spatial and spectral grid.

    probe_pump_idx : dict, optional
        Dictionary specifying which fields are probe and pump. Required keys are:
            - 'probe' : list of int
                Indices of the fields to be used as probe.
            - 'pump' : list of int
                Indices of the fields to be used as pump.
                
        If not provided, defaults to {"probe": [0], "pump": [1]}.

    fft_executor: quvac.pyfftw_executor.FFTExecutor, optional
        Executor that performs FFTs.
    nthreads : int, optional
        Number of threads to use for calculations. If not provided, defaults to the 
        number of CPU cores.

    Attributes
    ----------
    probe_pump_idx : dict
        Dictionary specifying which fields are probe and pump.
    nthreads : int
        Number of threads to use for calculations.
    probe_field : ExternalField
        ExternalField object for the probe fields.
    pump_field : ExternalField
        ExternalField object for the pump fields.
    """

    def __init__(self, fields_params, grid, probe_pump_idx=None, 
                 fft_executor=None, nthreads=None):
        if not probe_pump_idx:
            probe_pump_idx = {"probe": [0], "pump": [1]}
        self.probe_pump_idx = probe_pump_idx

        probe_params = [fields_params[idx] for idx in probe_pump_idx["probe"]]
        pump_params = [fields_params[idx] for idx in probe_pump_idx["pump"]]

        self.probe_field = ExternalField(probe_params, grid, fft_executor, 
                                         nthreads=nthreads)
        self.pump_field = ExternalField(pump_params, grid, fft_executor, 
                                        nthreads=nthreads)

    def calculate_field(self, t, E_probe=None, B_probe=None, E_pump=None, B_pump=None):
        """
        Calculate the probe and pump fields at a given time t.

        Parameters
        ----------
        t : float
            The time at which to calculate the fields.
        E_probe : np.array, optional
            Array to store the electric field output for the probe.
        B_probe : np.array, optional
            Array to store the magnetic field output for the probe.
        E_pump : np.array, optional
            Array to store the electric field output for the pump.
        B_pump : np.array, optional
            Array to store the magnetic field output for the pump.

        Returns
        -------
        tuple of tuple of np.array
            The calculated electric and magnetic fields for the probe and pump.
        """
        self.probe_field.calculate_field(t, E_out=E_probe, B_out=B_probe)
        self.pump_field.calculate_field(t, E_out=E_pump, B_out=B_pump)
        return (E_probe, B_probe), (E_pump, B_pump)
