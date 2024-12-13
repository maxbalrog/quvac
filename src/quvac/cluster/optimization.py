#!/usr/bin/env python3
"""
Script to run Bayesian optimization on cluster with Slurm
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from submitit import AutoExecutor, DebugJob, LocalJob

from quvac.cluster.config import DEFAULT_SUBMITIT_PARAMS
from quvac.simulation import quvac_simulation
from quvac.utils import read_yaml, write_yaml


def prepare_params_for_ax(params, ini_file):
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


def quvac_evaluation(params):
    ini_data = read_yaml(params["ini_default"])
    params.pop("ini_default")
    trial_idx = params.pop("trial_idx")
    scales = ini_data.get("scales", {})

    # Create ini.yml file for current trial
    trial_str = str(trial_idx).zfill(3)
    save_folder = ini_data["save_path"]
    save_path = os.path.join(save_folder, trial_str)

    # Update_parameters for current trial
    for param_key, param in params.items():
        category, key = param_key.split(":")
        scale = scales.get(key, 1)
        ini_data["fields"][category][key] = float(param * scale)

    # Save ini file
    ini_path = os.path.join(save_path, "ini.yml")
    Path(os.path.dirname(ini_path)).mkdir(parents=True, exist_ok=True)
    write_yaml(ini_path, ini_data)

    # Run simulation
    quvac_simulation(ini_path)

    # Load results
    data = np.load(os.path.join(save_path, "spectra.npz"))
    N_disc = data.get("N_disc", 0)
    N_total = data.get("N_total", 0)

    metrics = {
        "N_disc": (float(N_disc), 0.0),
        "N_total": (float(N_total), 0.0),
    }
    return metrics


def run_optimization(ax_client, executor, n_trials, max_parallel_jobs, experiment_file):
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


def parse_args():
    description = "Perform optimization of quvac simulations"
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument(
        "--input", "-i", default=None, help="Input yaml file with field and grid params"
    )
    argparser.add_argument(
        "--output", "-o", default=None, help="Path to save simulation data to"
    )
    argparser.add_argument(
        "--optimization", default=None, help="Yaml file with optimization parameters"
    )
    argparser.add_argument(
        "--wisdom", default="wisdom/fftw-wisdom", help="File to save pyfftw-wisdom"
    )
    return argparser.parse_args()


def cluster_optimization(ini_file, optimization_file, save_path=None, wisdom_file=None):
    """
    Launch optimization of quvac simulation for given default <ini>.yml file
    and <variables>.yml file

    Parameters:
    -----------
    ini_file: str (format <path>/<file_name>.yaml)
        Default initial configuration file containing all simulation parameters.
        These parameters (apart from variables) would remain the same for the
        whole gridscan.
        Note: This file might also contain parameters for cluster computation
    variables_file: str (format <path>/<file_name>.yaml)
        File containing all parameters to vary.
        Note (different from gridscan script): this file should contain only field
        parameters
    save_path: str
        Path to save simulation results to
    """
    # Check that ini file and save_path exists
    err_msg = f"{ini_file} or {optimization_file} is not a file or does not exist"
    assert os.path.isfile(ini_file) and os.path.isfile(optimization_file), err_msg
    if save_path is None:
        save_path = os.path.dirname(ini_file)
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
    experiment_file = os.path.join(save_path, "experiment.json")

    ini_default = read_yaml(ini_file)
    optimization_params = read_yaml(optimization_file)
    cluster_params = optimization_params.get("cluster", {})

    # Check that optimization parameters are only field_parameters
    optimization_keys = optimization_params["parameters"].keys()
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
        # executor.update_parameters(slurm_array_parallelism=max_parallel_jobs)
        executor.update_parameters(**sbatch_params)

    n_trials = optimization_params.get("n_trials", 10)

    run_optimization(ax_client, executor, n_trials, max_parallel_jobs, experiment_file)
    print("Optimization finished!")


def gather_trials_data(ax_client, metric_names=["N_total", "N_disc"]):
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
    args = parse_args()
    cluster_optimization(args.input, args.optimization, args.output)
