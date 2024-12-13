#!/usr/bin/env python3
"""
Script to run gridscan simulations on cluster with Slurm
"""
import argparse
import itertools
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import submitit

from quvac.cluster.config import DEFAULT_SUBMITIT_PARAMS
from quvac.simulation import quvac_simulation
from quvac.utils import read_yaml, write_yaml


def _create_grids(variables):
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
    Dictionary key 'fields' is handled separately because it's a 3-level dict
    (fields: key_1: key_2: value) while other parameters are 2-level dicts
    (key_1: key_2: value)
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
    Transform nested dict into plane dict by combining nested
    dict keys
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
    Creates separate ini.yml file for every combination of params
    in gridscan
    """
    ini_files = []
    for parametrization in itertools.product(*param_grids):
        ini_current = deepcopy(ini_default)
        name_local = ""
        for name, param in zip(param_names, parametrization):
            category, param_name = name
            if category.startswith("field"):
                ini_current["fields"][category][param_name] = float(param)
            else:
                ini_current[category][param_name] = float(param)

            param = int(param) if np.isclose(param, int(param)) else param
            param_str = str(param) if isinstance(param, int) else f"{param:.2f}"
            param_str = f"{category}:{param_name}_{param_str}"
            name_local = "#".join([name_local, param_str])
        save_path_local = os.path.join(save_path, name_local, "ini.yml")
        Path(os.path.dirname(save_path_local)).mkdir(parents=True, exist_ok=True)
        write_yaml(save_path_local, ini_current)
        ini_files.append(save_path_local)
    return ini_files


def parse_args():
    description = "Perform gridscan of quvac simulations"
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument(
        "--input", "-i", default=None, help="Input yaml file with field and grid params"
    )
    argparser.add_argument(
        "--output", "-o", default=None, help="Path to save simulation data to"
    )
    argparser.add_argument(
        "--variables", default=None, help="Yaml file with variable parameters"
    )
    argparser.add_argument(
        "--wisdom", default="wisdom/fftw-wisdom", help="File to save pyfftw-wisdom"
    )
    return argparser.parse_args()


def cluster_gridscan(ini_file, variables_file, save_path=None, wisdom_file=None):
    """
    Launch a gridscan of quvac simulation for given default <ini>.yml file
    and <variables>.yml file

    Parameters:
    -----------
    ini_file: str (format <path>/<file_name>.yaml)
        Default initial configuration file containing all simulation parameters.
        These parameters (apart from variables) would remain the same for the
        whole gridscan.
        Note: This file might also contain parameters for cluster computation
    variables_file: str (format <path>/<file_name>.yaml)
        File containing all parameters to vary
    save_path: str
        Path to save simulation results to
    """
    # Check that ini file and save_path exists
    err_msg = f"{ini_file} or {variables_file} is not a file or does not exist"
    assert os.path.isfile(ini_file) and os.path.isfile(variables_file), err_msg
    if save_path is None:
        save_path = os.path.dirname(ini_file)
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)

    ini_default = read_yaml(ini_file)
    variables = read_yaml(variables_file)
    cluster_params = variables.get("cluster", {})
    if cluster_params:
        variables.pop("cluster")

    create_grids = variables.get("create_grids", False)
    if "create_grids" in variables:
        variables.pop("create_grids")

    # Create parameter grids if required
    if create_grids:
        variables_grid = create_parameter_grids(variables)
    else:
        variables_grid = variables

    # Restructure variables dict
    param_names, param_grids = restructure_variables_grid(variables_grid)

    # Create yaml files for the grid scan
    ini_files = create_ini_files_for_gridscan(
        ini_default, param_names, param_grids, save_path
    )

    # Set up scheduler
    cluster = cluster_params.get("cluster", "local")
    log_folder = os.path.join(save_path, "submitit_logs")
    sbatch_params = cluster_params.get("sbatch_params", DEFAULT_SUBMITIT_PARAMS)
    max_jobs = cluster_params.get("max_parallel_jobs", 5)
    executor = submitit.AutoExecutor(folder=log_folder, cluster=cluster)
    if cluster == "slurm":
        executor.update_parameters(slurm_array_parallelism=max_jobs)
        executor.update_parameters(**sbatch_params)
    else:
        executor.update_parameters(timeout_min=30)
    print("Submitting jobs...")
    jobs = executor.map_array(quvac_simulation, ini_files)
    print("Jobs submitted, waiting for results...")

    # Wait till all jobs end
    outputs = [job.result() for job in jobs]
    print("Grid scan is finished!")


if __name__ == "__main__":
    args = parse_args()
    cluster_gridscan(args.input, args.variables, args.output)
