#!/usr/bin/env python3
"""
Script to run Bayesian optimization on cluster with Slurm.

Optimization parameters (optimized variables, objectives and constraints)
should be located in `ini.yml` at key `optimization`.

Usage:

.. code-block:: bash

    optimization.py -i <input>.yaml -o <output_dir>
"""
import os
import time
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from submitit import AutoExecutor, DebugJob, LocalJob

from quvac.cluster.config import DEFAULT_SUBMITIT_PARAMS
from quvac.postprocess import signal_in_detector, integrate_spherical
from quvac.simulation import quvac_simulation, parse_args
from quvac.utils import read_yaml, write_yaml, round_to_n


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
    detectors = obj_params["detectors"]
    detectors = [detectors] if isinstance(detectors, dict) else detectors
    k, theta, phi, N_sph = [data[key] for key in "k theta phi N_sph".split()]
    N_angular = integrate_spherical(N_sph, (k,theta,phi), axs_integrate=['k'])

    N_detector = 0
    for detector in detectors:
        N_detector += signal_in_detector(N_angular, theta, phi, detector,
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
               "exactly one more field than free parameters")
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


def collect_metrics(data, obj_params, metric_names=["N_total"]):
    """
    Collect metrics from simulation results.

    Parameters
    ----------
    data : dict
        Dictionary containing simulation results.
    obj_params : dict
        Dictionary containing objective parameters.
    metric_names : list of str, optional
        List of metric names to collect. Default is ["N_total"].

    Returns
    -------
    dict
        Dictionary containing the collected metrics.
    """
    N_disc = data.get("N_disc", 0)
    N_total = data.get("N_total", 0)

    metrics = {}
    metrics["N_total"] = (float(N_total), 0.0)
    if N_disc is not None:
        metrics["N_disc"] = (float(N_disc), 0.0)

    if "detectors" in obj_params:
        N_detector = objective_signal_in_detector(data, obj_params)
        metrics["N_detector"] = (float(N_detector), 0.0)

    # filter metrics
    metrics = {k:v for k,v in metrics.items() if k in metric_names}
    return metrics


def quvac_evaluation(params, metric_names=["N_total"]):
    """
    Evaluate a single trial of the quvac simulation.

    Parameters
    ----------
    params : dict
        Dictionary containing trial parameters.
    metric_names : list of str, optional
        List of metric names to evaluate. Default is ["N_total"].

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
    metrics = collect_metrics(data, obj_params, metric_names)
    return metrics


def check_energy_constraint(trial_index_to_param):
    """
    Check if the total energy budget constraint is satisfied.

    Parameters
    ----------
    trial_index_to_param : dict
        Dictionary mapping trial indices to parameter dictionaries. Each parameter dictionary
        contains the energy distribution among fields.

    Returns
    -------
    bool
        True if the total energy budget constraint is satisfied for all trials, False otherwise.

    Raises
    ------
    Warning
        If the total energy budget constraint is violated for any trial.
    """
    energy_ok = True
    for trial_idx, params in trial_index_to_param.items():
        energies = []
        for param_key, param in params.items():
            if param_key != "ini_default":
                category, key = param_key.split(":")
                if key == "W":
                    energies.append(param)
        
        W_total = np.sum(energies)
        eps = 1.0 - W_total
        if eps < 0 and not np.isclose(abs(eps), 0.0, atol=1e-5):
            warnings.warn("Fixed total energy budget constraint is violated! "
                          "Probably, optimization fails to find new prospective points and"
                          "is stuck in local minima.")
            energy_ok = False
            break
    return energy_ok


def check_repeated_samples(trial_index_to_param, last_samples, fail_count, patience=3):
    """
    Check for repeated samples in the optimization process.

    Parameters
    ----------
    trial_index_to_param : dict
        Dictionary mapping trial indices to parameter dictionaries.
    last_samples : list of tuple
        List of parameter tuples from the most recent trials.
    fail_count : int
        Counter for the number of consecutive repeated samples.
    patience : int, optional
        Maximum number of consecutive repeated samples allowed before stopping. Default is 3.

    Returns
    -------
    tuple
        A tuple containing:
        - continue_sampling (bool): Whether to continue sampling new trials.
        - last_samples (list of tuple): Updated list of parameter tuples.
        - fail_count (int): Updated counter for repeated samples.

    Notes
    -----
    If the number of repeated samples exceeds the `patience` value, a warning is issued,
    and sampling is stopped.
    """
    continue_sampling = True
    trials = deepcopy(trial_index_to_param)
    for trial_idx, params in trials.items():
        params.pop("ini_default")
        # round-up ints and floats for comparison
        params_list = []
        for k,v in tuple(sorted(params.items())):
            if (isinstance(v, int) or isinstance(v, float)) and not np.isclose(v, 0.0):
                # we use 5 significant digits
                v = round_to_n(v,5)
            params_list.append((k,v))
        params_tuple = tuple(params_list)

        if last_samples and params_tuple == last_samples[-1]:
            fail_count += 1
            warnings.warn(f"Trial {len(last_samples)-1} is identical to previous: {params}. Fail count: {fail_count}")
        else:
            fail_count = 0
        
        last_samples.append(params_tuple)

        if fail_count >= patience:
            warnings.warn(f"Number of repeated samples exceeded patience ({patience} tries)!")
            continue_sampling = False
    return continue_sampling, fail_count


def check_sampled_trials(trial_index_to_param, last_samples, fail_count):
    """
    Check if the sampled trials are valid.

    Parameters
    ----------
    trial_index_to_param : dict
        Dictionary mapping trial indices to parameter dictionaries.
    last_samples : list of tuple
        List of parameter tuples from the most recent trials.
    fail_count : int
        Counter for the number of consecutive repeated samples.

    Returns
    -------
    tuple
        A tuple containing:
        - continue_optimization (bool): Whether to continue the optimization process.
        - last_samples (list of tuple): Updated list of parameter tuples.
        - fail_count (int): Updated counter for repeated samples.

    Notes
    -----
    This function checks two conditions:
    1. Whether the total energy budget constraint is satisfied.
    2. Whether there are repeated samples in the optimization process.
    If either condition fails, the optimization process is terminated.
    """
    energy_ok = check_energy_constraint(trial_index_to_param)
    continue_sampling, fail_count = check_repeated_samples(trial_index_to_param, last_samples, fail_count)

    continue_optimization = energy_ok and continue_sampling
    return continue_optimization, fail_count


def run_optimization(ax_client, executor, n_trials, max_parallel_jobs, experiment_file,
                     metric_names=["N_total"]):
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
    metric_names : list of str, optional
        List of metric names to evaluate. Default is ["N_total"].

    Returns
    -------
    None
    """
    jobs = []
    submitted_jobs = 0
    # variables for early optimization stopping
    continue_optimization = True
    last_samples = []
    fail_count = 0
    # Run until all the jobs have finished and our budget is used up.
    while (continue_optimization and submitted_jobs < n_trials) or jobs:
        for job, trial_idx in jobs:
            # Poll if any jobs completed
            # Local and debug jobs don't run until .result() is called.
            if job.done() or type(job) in [LocalJob, DebugJob]:
                result = job.result()
                ax_client.complete_trial(trial_index=trial_idx, raw_data=result)
                jobs.remove((job, trial_idx))
                ax_client.save_to_json_file(experiment_file)

        # sample new points if optimization is not terminated
        if continue_optimization:
            # Schedule new jobs if there is availablity
            trial_index_to_param, _ = ax_client.get_next_trials(
                max_trials=min(max_parallel_jobs - len(jobs), n_trials - submitted_jobs)
            )
            # Check that sampled trials satisfy the requirements
            if trial_index_to_param:
                continue_optimization, fail_count = check_sampled_trials(trial_index_to_param,
                                                                         last_samples,
                                                                         fail_count)

                if continue_optimization:
                    for trial_idx, params in trial_index_to_param.items():
                        params["trial_idx"] = trial_idx
                        job = executor.submit(quvac_evaluation, params, metric_names)
                        submitted_jobs += 1
                        jobs.append((job, trial_idx))
                        time.sleep(1)
                else:
                    warnings.warn("Terminating optimization... Finishing last trials...")
                

def setup_generation_strategy(num_random_trials=6):
    """
    Setup custom generation strategy.

    Parameters
    ----------
    num_random_trials : int, optional
        Number of random trials to perform before starting Bayesian optimization. Default is 6.

    Returns
    -------
    ax.modelbridge.generation_strategy.GenerationStrategy
        The configured generation strategy.
    """
    gs = GenerationStrategy(
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=num_random_trials),
            GenerationStep(model=Models.GPEI, num_trials=-1),
        ]
    )
    return gs


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

    # custom generation strategy
    gs_params = optimization_params.get("gs_params", {})
    num_random_trials = gs_params.get("num_random_trials", 6)
    gs = setup_generation_strategy(num_random_trials=num_random_trials)

    # Set up optimization client
    ax_client = AxClient(generation_strategy=gs)
    objectives = optimization_params["objectives"]
    track_metrics = optimization_params.get("track_metrics", [])
    # collect metric names to keep track of
    metric_names = [name for name, flag in objectives]
    metric_names += track_metrics

    ax_client.create_experiment(
        name=optimization_params.get("name", "test_optimization"),
        parameters=params_for_ax,
        objectives={
            name: ObjectiveProperties(minimize=flag) for name, flag in objectives
        },
        parameter_constraints=optimization_params.get("parameter_constraints", None),
        outcome_constraints=optimization_params.get(
            "outcome_constraints", None
        ),
        tracking_metric_names=track_metrics,
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

    run_optimization(ax_client, executor, n_trials, max_parallel_jobs, experiment_file,
                     metric_names)
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
    trials_params = {key: trial.arm.parameters for key, trial in trials.items()
                     if trial.completed_successfully}
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
    args = parse_args(description="Perform optimization of quvac simulations")
    cluster_optimization(args.input, args.output)
