#!/usr/bin/env python3
"""
Script to run Bayesian optimization on cluster with Slurm.

Optimization parameters (optimized variables, objectives and constraints)
should be located in `ini.yml` at key `optimization`.

Usage:

.. code-block:: bash

    optimization.py -i <input>.yaml -o <output_dir>
"""
import argparse
import os
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from submitit import AutoExecutor, DebugJob, LocalJob

from quvac.cluster.config import DEFAULT_SUBMITIT_PARAMS
from quvac.postprocess import signal_in_detector, integrate_spherical
from quvac.simulation import quvac_simulation
from quvac.utils import read_yaml, write_yaml


def prepare_params_for_ax(params, ini_file):
    """
    Prepare parameters for Ax optimization.

    Parameters
    ----------
    params : dict
        Dictionary containing parameter categories and their bounds.
    ini_file : str
        Path to the initial configuration file.

    Returns
    -------
    list of dict
        List of parameter descriptions formatted for Ax optimization.

    Note
    ----
    Only field parameters could be optimized. They are given as dictionaries
    {"field_1": {"param_1": range_1, "param_2": range_2}, ...} which are transformed
    to ["field_1:param_1", "field_1:param_2", ...] parameter names.

    Also the path to default `ini.yml` is passed. For every optimization trial it is 
    loaded, optimized parameters are changed and the simulation is submitted.
    """
    params_ax = []
    for category_key, category in params.items():
        for key, val in category.items():
            loc_name = f"{category_key}:{key}"
            param_descr = {
                "name": loc_name,
                "type": "range",
                "bounds": val,
                "value_type": "float",
            }
            params_ax.append(param_descr)
    params_ax.append({"name": "ini_default", "type": "fixed", "value": ini_file})
    return params_ax


def objective_signal_in_detector(data, obj_params):
    """
    Calculate the signal detected within a specified detector region.

    Parameters
    ----------
    data : dict
        Dictionary containing simulation results, including spherical grid data.
    obj_params : dict
        Dictionary containing detector parameters.

    Returns
    -------
    float
        The signal detected within the specified detector region.
    """
    detector = obj_params["detector"]
    k, theta, phi, N_sph = [data[key] for key in "k theta phi N_sph".split()]
    N_angular = integrate_spherical(N_sph, (k,theta,phi), axs_integrate=['k'])
    N_detector = signal_in_detector(N_angular, theta, phi, detector,
                                    align_to_max=False)
    return N_detector


def update_energies(ini_data, energy_params):
    """
    Update the energy distribution among fields for optimization.

    Parameters
    ----------
    ini_data : dict
        Dictionary containing the initial configuration data.
    energy_params : dict
        Dictionary containing the energy parameters to optimize.

    Returns
    -------
    dict
        Updated configuration data with modified energy distribution.

    Raises
    ------
    AssertionError
        If the number of fields does not match the number of optimized parameters plus one.
    """
    ini = deepcopy(ini_data)
    optimization_params = ini["optimization"]
    energy_fields = optimization_params["energy_fields"]
    fields, opt_fields = [energy_fields[key] for key
                          in "fields optimized_fields".split()]
    err_msg = ("While optimizing energy distribution, it is required to have "
               "exactly one more fields than free parameters")
    assert len(fields) == len(opt_fields)+1, err_msg

    optimization_params = ini_data["optimization"]
    scales = optimization_params.get("scales", {})
    scale = scales.get("W", 1)

    W_total = 0.
    for param_key, param in energy_params.items():
        category, key = param_key.split(":")
        ini["fields"][category][key] = float(param * scale)
        W_total += param
    
    # fix energy of remaining field
    idx_remain = list(set(fields) - set(opt_fields))[0]
    W_remain = np.maximum(1. - W_total, 0.)
    ini["fields"][f"field_{idx_remain}"]["W"] = float(W_remain * scale)
    return ini


def quvac_evaluation(params):
    """
    Evaluate a single trial of the quvac simulation.

    Parameters
    ----------
    params : dict
        Dictionary containing trial parameters.

    Returns
    -------
    dict
        Dictionary containing metrics such as `N_disc`, `N_total`, and optionally `N_detector`.
    """
    ini_file = params["ini_default"]
    ini_data = read_yaml(ini_file)
    params.pop("ini_default")
    trial_idx = params.pop("trial_idx")

    optimization_params = ini_data["optimization"]
    obj_params = optimization_params.get("objectives_params", {})
    scales = optimization_params.get("scales", {})

    # Create ini.yml file for current trial
    trial_str = str(trial_idx).zfill(3)
    save_folder = ini_data.get("save_path", os.path.dirname(ini_file))
    save_path = os.path.join(save_folder, trial_str)

    energy_params = {}
    # Update_parameters for current trial
    for param_key, param in params.items():
        category, key = param_key.split(":")
        if key != "W":
            scale = scales.get(key, 1)
            ini_data["fields"][category][key] = float(param * scale)
        else:
            energy_params[param_key] = param

    # treat separately energy distribution parameters
    if energy_params:        
        ini_data = update_energies(ini_data, energy_params)

    # Save ini file
    ini_path = os.path.join(save_path, "ini.yml")
    Path(os.path.dirname(ini_path)).mkdir(parents=True, exist_ok=True)
    write_yaml(ini_path, ini_data)

    # Run simulation
    quvac_simulation(ini_path)

    # Load results
    data = np.load(os.path.join(save_path, "spectra_total.npz"))
    N_disc = data.get("N_disc", 0)
    N_total = data.get("N_total", 0)

    metrics = {
        "N_disc": (float(N_disc), 0.0),
        "N_total": (float(N_total), 0.0),
    }
    if "detector" in obj_params:
        N_detector = objective_signal_in_detector(data, obj_params)
        metrics["N_detector"] = (float(N_detector), 0.0)
    return metrics


def run_optimization(ax_client, executor, n_trials, max_parallel_jobs, experiment_file):
    """
    Run Bayesian optimization using Ax and Submitit.

    Parameters
    ----------
    ax_client : ax.service.ax_client.AxClient
        Ax client for managing the optimization process.
    executor : submitit.AutoExecutor
        Submitit executor for running jobs on a cluster.
    n_trials : int
        Total number of trials to run.
    max_parallel_jobs : int
        Maximum number of parallel jobs to run.
    experiment_file : str
        Path to save the Ax experiment data.

    Returns
    -------
    None
    """
    jobs = []
    submitted_jobs = 0
    # Run until all the jobs have finished and our budget is used up.
    while submitted_jobs < n_trials or jobs:
        for job, trial_idx in jobs:
            # Poll if any jobs completed
            # Local and debug jobs don't run until .result() is called.
            if job.done() or type(job) in [LocalJob, DebugJob]:
                result = job.result()
                ax_client.complete_trial(trial_index=trial_idx, raw_data=result)
                jobs.remove((job, trial_idx))
                ax_client.save_to_json_file(experiment_file)

        # Schedule new jobs if there is availablity
        trial_index_to_param, _ = ax_client.get_next_trials(
            max_trials=min(max_parallel_jobs - len(jobs), n_trials - submitted_jobs)
        )
        for trial_idx, params in trial_index_to_param.items():
            params["trial_idx"] = trial_idx
            job = executor.submit(quvac_evaluation, params)
            submitted_jobs += 1
            jobs.append((job, trial_idx))
            time.sleep(1)


def _parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    description = "Perform optimization of quvac simulations"
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument(
        "--input", "-i", default=None, help="Input yaml file with field and grid params"
    )
    argparser.add_argument(
        "--output", "-o", default=None, help="Path to save simulation data to"
    )
    argparser.add_argument(
        "--wisdom", default="wisdom/fftw-wisdom", help="File to save pyfftw-wisdom"
    )
    return argparser.parse_args()


def cluster_optimization(ini_file, save_path=None, wisdom_file=None):
    """
    Launch optimization of quvac simulation for a given initial configuration file.

    Parameters
    ----------
    ini_file : str
        Path to the initial configuration file (YAML format).
    save_path : str, optional
        Path to save simulation results. If not provided, defaults to the directory of `ini_file`.
    wisdom_file : str, optional
        Path to save FFTW wisdom. Default is None.

    Returns
    -------
    None
    """
    # Check that ini file and save_path exists
    assert os.path.isfile(ini_file), f"{ini_file} is not a file or does not exist"
    if save_path is None:
        save_path = os.path.dirname(ini_file)
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
    experiment_file = os.path.join(save_path, "experiment.json")

    ini_default = read_yaml(ini_file)
    optimization_params = ini_default["optimization"]
    cluster_params = optimization_params.get("cluster", {})

    # Check that optimization parameters are only field_parameters
    optimization_keys = list(optimization_params["parameters"].keys())
    err_msg = (
        f"Only field parameters could be optimized but you have {optimization_keys}"
    )
    assert sum([key.startswith("field") for key in optimization_keys]) == len(
        optimization_keys
    ), err_msg

    # Prepare parameters for optimization in ax style
    params_for_ax = prepare_params_for_ax(optimization_params["parameters"], ini_file)

    # Set up optimization client
    ax_client = AxClient()
    objectives = optimization_params["objectives"]

    ax_client.create_experiment(
        name=optimization_params.get("name", "test_optimization"),
        parameters=params_for_ax,
        objectives={
            name: ObjectiveProperties(minimize=flag) for name, flag in objectives
        },
        parameter_constraints=optimization_params.get("parameter_constraints", None),
        outcome_constraints=optimization_params.get(
            "outcome_constraints", None
        ),  # Optional.
    )

    # Set up sibmitit AutoExecutor
    cluster = cluster_params.get("cluster", "local")
    sbatch_params = cluster_params.get("sbatch_params", DEFAULT_SUBMITIT_PARAMS)
    max_parallel_jobs = cluster_params.get("max_parallel_jobs", 3)
    log_folder = os.path.join(save_path, "submitit_logs")

    executor = AutoExecutor(folder=log_folder, cluster=cluster)
    if cluster == "slurm":
        executor.update_parameters(slurm_array_parallelism=max_parallel_jobs)
        executor.update_parameters(**sbatch_params)
    elif "timeout_min" in cluster_params:
        executor.update_parameters(timeout_min=cluster_params["timeout_min"])

    n_trials = optimization_params.get("n_trials", 10)

    run_optimization(ax_client, executor, n_trials, max_parallel_jobs, experiment_file)
    print("Optimization finished!")


def gather_trials_data(ax_client, metric_names=["N_total", "N_disc"]):
    """
    Gather data from completed trials in the Ax experiment.

    Parameters
    ----------
    ax_client : ax.service.ax_client.AxClient
        Ax client managing the optimization process.
    metric_names : list of str, optional
        List of metric names to extract from the trials. Default is ["N_total", "N_disc"].

    Returns
    -------
    dict
        Dictionary containing trial parameters and their corresponding metrics.
    """
    metrics = ax_client.experiment.fetch_data().df
    trials = ax_client.experiment.trials
    trials_params = {key: trial.arm.parameters for key, trial in trials.items()}
    for key in trials_params:
        for metric_name in metric_names:
            condition = (metrics["metric_name"] == metric_name) & (
                metrics["trial_index"] == key
            )
            trials_params[key][metric_name] = metrics.loc[condition]["mean"].iloc[0]
        if "ini_default" in trials_params[key]:
            trials_params[key].pop("ini_default")
    return trials_params


if __name__ == "__main__":
    args = _parse_args()
    cluster_optimization(args.input, args.output)
