#!/usr/bin/env python3
"""
Script to run gridscan simulations on cluster with Slurm.

Gridscan parameters (scanned variables and their grids)
should be located in `ini.yml` at key `gridscan`.

Usage:

.. code-block:: bash

    gridscan.py -i <input>.yml -o <output_dir>
"""
from copy import deepcopy
import itertools
import logging
import os
from pathlib import Path

import numpy as np

from quvac.log import log_time
from quvac.parallel import run_simulations_with_job_executor
from quvac.simulation import create_basic_logger, get_dirs, parse_args
from quvac.utils import read_yaml, write_yaml

_logger = logging.getLogger("simulation")


def _create_grids(variables):
    """
    Create parameter grids for each category and parameter.

    Parameters
    ----------
    variables : dict
        Dictionary containing parameter categories and their bounds. Each parameter
        should be specified as a tuple (start, end, npts).

    Returns
    -------
    dict
        Dictionary containing parameter grids for each category and parameter.
    """
    variables_grid = {}
    for category_key, category in variables.items():
        variables_grid[category_key] = {}
        for param_key, param in category.items():
            start, end, npts = param
            param_grid = list(np.linspace(start, end, npts))
            variables_grid[category_key][param_key] = param_grid
    return variables_grid


def create_parameter_grids(variables):
    """
    Create a grid from (start, end, npts) specified for each parameter.

    Parameters
    ----------
    variables : dict
        Dictionary containing parameter categories and their bounds. The key 'fields'
        is handled separately as it is a 3-level dictionary.

    Returns
    -------
    dict
        Dictionary containing parameter grids for all categories and parameters.
    """
    variables_grid = {}
    fields = variables.get("fields", {})
    if fields:
        fields_grid = _create_grids(fields)
        variables_grid["fields"] = fields_grid
        variables.pop("fields")
    params_grid = _create_grids(variables)
    variables_grid.update(params_grid)
    return variables_grid


def restructure_variables_grid(variables):
    """
    Transform a nested dictionary into a flat dictionary by combining nested keys.

    Parameters
    ----------
    variables : dict
        Dictionary containing parameter grids for each category and parameter.

    Returns
    -------
    tuple
        A tuple containing:
        - param_names : list of list of str
            List of parameter names split into categories and parameter keys.
        - param_grids : list of list
            List of parameter grids corresponding to each parameter.
    """
    variables_grid = {}
    for key, val in variables["fields"].items():
        variables[key] = val
    variables.pop("fields")

    for category_key, category in variables.items():
        for param_key, param in category.items():
            new_key = f"{category_key}:{param_key}"
            variables_grid[new_key] = param

    param_names = list(variables_grid.keys())
    param_names = [param.split(":") for param in param_names]
    param_grids = list(variables_grid.values())
    return param_names, param_grids


def create_ini_files_for_gridscan(ini_default, param_names, param_grids, save_path):
    """
    Create separate `ini.yml` files for every combination of parameters in the 
    grid scan.

    Parameters
    ----------
    ini_default : dict
        Default initial configuration dictionary containing all simulation parameters.
    param_names : list of list of str
        List of parameter names split into categories and parameter keys.
    param_grids : list of list
        List of parameter grids corresponding to each parameter.
    save_path : str
        Path to save the generated `ini.yml` files.

    Returns
    -------
    list of str
        List of file paths to the generated `ini.yml` files.
    """
    ini_files = []
    for parametrization in itertools.product(*param_grids):
        ini_current = deepcopy(ini_default)
        name_local = ""
        for name, param in zip(param_names, parametrization, strict=True):
            category, param_name = name
            if category.startswith("field"):
                ini_current["fields"][category][param_name] = float(param)
            else:
                ini_current[category][param_name] = float(param)

            param = int(param) if np.isclose(param, int(param)) else param
            param_str = str(param) if isinstance(param, int) else f"{param:.2e}"
            param_str = f"{category}:{param_name}_{param_str}"
            name_local = "#".join([name_local, param_str])
        save_path_local = os.path.join(save_path, name_local, "ini.yml")
        Path(os.path.dirname(save_path_local)).mkdir(parents=True, exist_ok=True)
        write_yaml(save_path_local, ini_current)
        ini_files.append(save_path_local)
    return ini_files


def cluster_gridscan(ini_file, save_path=None, wisdom_file=None):
    """
    Launch a grid scan of quvac simulations for a given default `ini.yml` file.

    Parameters
    ----------
    ini_file : str
        Path to the default initial configuration file (YAML format) containing all 
        simulation parameters.
    save_path : str, optional
        Path to save simulation results. If not provided, defaults to the directory of 
        `ini_file`.
    wisdom_file : str, optional
        Path to save FFTW wisdom. Default is None.

    Returns
    -------
    None
    """
    # Check that ini file and save_path exist
    files = get_dirs(ini_file, save_path, wisdom_file, mode="gridscan")
    save_path = files['save_path']

    # Setup logger
    create_basic_logger(files["logger"])

    # Start time
    log_time(_logger, name="start")

    ini_default = read_yaml(ini_file)
    gridscan_params = ini_default["gridscan"]
    cluster_params = gridscan_params.get("cluster_params", {})
    if cluster_params:
        gridscan_params.pop("cluster_params")

    create_grids = gridscan_params.get("create_grids", False)
    if "create_grids" in gridscan_params:
        gridscan_params.pop("create_grids")

    # Create parameter grids if required
    if create_grids:
        variables_grid = create_parameter_grids(gridscan_params)
    else:
        variables_grid = gridscan_params

    # Restructure variables dict
    param_names, param_grids = restructure_variables_grid(variables_grid)

    # Create yaml files for the grid scan
    ini_files = create_ini_files_for_gridscan(
        ini_default, param_names, param_grids, save_path
    )
    _logger.info("MILESTONE: ini files are created for the specified grid.")

    run_simulations_with_job_executor(
        ini_files, 
        cluster_params, 
        save_path,
        max_parallel_jobs_default=5,
    )
    _logger.info("Grid scan is finished!")

    # End time
    log_time(_logger, name="end")


def main_gridscan():
    """
    Main function to run the gridscan.
    """
    args = parse_args(description="Perform gridscan of quvac simulations")
    cluster_gridscan(args.input, args.output)


if __name__ == "__main__":
    main_gridscan()
