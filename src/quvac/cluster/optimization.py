#!/usr/bin/env python3
"""
Script to run Bayesian optimization on cluster with Slurm.

Optimization parameters (optimized variables, objectives and constraints)
should be located in `ini.yml` at key `optimization`.

Usage:

.. code-block:: bash

    optimization.py -i <input>.yml -o <output_dir>
"""
from collections.abc import Iterable
from copy import deepcopy
import itertools
import logging
import os
from pathlib import Path
import time
import warnings

from ax.adapter.registry import Generators
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.observation import ObservationFeatures
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
import numpy as np
from submitit import DebugJob, LocalJob

from quvac.log import log_time
from quvac.parallel import setup_job_executor_from_params
from quvac.postprocess import integrate_spherical, signal_in_detector
from quvac.simulation import create_basic_logger, get_dirs, parse_args, quvac_simulation
from quvac.utils import read_yaml, round_to_n, write_yaml

_logger = logging.getLogger("simulation")


def prepare_params_for_ax(params):
    """
    Prepare parameters for Ax optimization.

    Parameters
    ----------
    params : dict
        Dictionary containing parameter categories and their bounds.

    Returns
    -------
    list of ax.api.configs.RangeParameterConfig
        List of parameter descriptions formatted for Ax optimization.

    Note
    ----
    Only field parameters could be optimized. They are given as dictionaries
    {"field_1": {"param_1": range_1, "param_2": range_2}, ...} which are transformed
    to ["field_1:param_1", "field_1:param_2", ...] parameter names.

    For every optimization trial it is loaded, optimized parameters are changed and 
    the simulation is submitted.
    """
    params_ax = []
    for category_key, category in params.items():
        for key, val in category.items():
            loc_name = f"{category_key}:{key}"
            param_descr = {
                "name": loc_name,
                "parameter_type": "float",
                "bounds": val,
            }
            params_ax.append(RangeParameterConfig(**param_descr))
    return params_ax


def objective_signal_in_detector(data, obj_params, discernible=False):
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
    
    N_detector_disc = 0
    if discernible:
        discernible_mask = data["discernible"]
        for detector in detectors:
            N_detector_disc += signal_in_detector(
                N_angular*discernible_mask, theta, phi, detector, align_to_max=False
            )
    return (N_detector, N_detector_disc)


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
        If the number of fields does not match the number of optimized parameters plus 
        one.
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


def collect_metrics(data, obj_params, metric_names=("N_total"), 
                    noiseless_observations=True):
    """
    Collect metrics from simulation results.

    Parameters
    ----------
    data : dict
        Dictionary containing simulation results.
    obj_params : dict
        Dictionary containing objective parameters.
    metric_names : list of str, optional
        List of metric names to collect. Default is ("N_total").

    Returns
    -------
    dict
        Dictionary containing the collected metrics.
    """
    N_disc = data.get("N_disc", 0)
    N_total = data.get("N_total", 0)

    metrics = {}
    metrics["N_total"] = float(N_total)
    if N_disc is not None:
        metrics["N_disc"] = float(N_disc)

    if "detectors" in obj_params:
        N_detector, N_detector_disc = objective_signal_in_detector(
            data, obj_params, discernible=True
        )
        metrics["N_detector"] = float(N_detector)
        metrics["N_detector_disc"] = float(N_detector_disc)

    # filter metrics
    if noiseless_observations:
        metrics = {k:(v,0.0) for k,v in metrics.items() if k in metric_names}
    else:
        metrics = {k:v for k,v in metrics.items() if k in metric_names}
    return metrics


def quvac_evaluation(params, trial_index, default_ini_file, metric_names=("N_total"),
                     noiseless_observations=True):
    """
    Evaluate a single trial of the quvac simulation.

    Parameters
    ----------
    params : dict
        Dictionary containing trial parameters.
    metric_names : list of str, optional
        List of metric names to evaluate. Default is ("N_total").

    Returns
    -------
    dict
        Dictionary containing metrics such as `N_disc`, `N_total`, and optionally 
        `N_detector`.
    """
    ini_data = read_yaml(default_ini_file)

    optimization_params = ini_data["optimization"]
    obj_params = optimization_params.get("objectives_params", {})
    scales = optimization_params.get("scales", {})

    # Create ini.yml file for current trial
    trial_str = str(trial_index).zfill(3)
    save_folder = ini_data.get("save_path", os.path.dirname(default_ini_file))
    save_path = os.path.join(save_folder, trial_str)

    energy_params = {}
    # Update parameters for current trial
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
    metrics = collect_metrics(data, obj_params, metric_names, noiseless_observations)
    write_yaml(os.path.join(save_path, "metrics.yml"), metrics)
    return metrics


def check_energy_constraint(trial_index_to_param):
    """
    Check if the total energy budget constraint is satisfied.

    Parameters
    ----------
    trial_index_to_param : dict
        Dictionary mapping trial indices to parameter dictionaries. Each parameter 
        dictionary contains the energy distribution among fields.

    Returns
    -------
    bool
        True if the total energy budget constraint is satisfied for all trials, False 
        otherwise.

    Raises
    ------
    Warning
        If the total energy budget constraint is violated for any trial.
    """
    energy_ok = True
    for _, params in trial_index_to_param.items():
        energies = []
        for param_key, param in params.items():
            category, key = param_key.split(":")
            if key == "W":
                energies.append(param)
        
        W_total = np.sum(energies)
        eps = 1.0 - W_total
        if eps < 0 and not np.isclose(abs(eps), 0.0, atol=1e-5):
            warnings.warn("Fixed total energy budget constraint is violated! "
                          "Probably, optimization fails to find new prospective points "
                          "and is stuck in local minima.", stacklevel=2)
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
        Maximum number of consecutive repeated samples allowed before stopping. Default 
        is 3.

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
    for _, params in trials.items():
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
            warnings.warn(f"Trial {len(last_samples)-1} is identical to previous: "
                          f"{params}. Fail count: {fail_count}", stacklevel=2)
        else:
            fail_count = 0
        
        last_samples.append(params_tuple)

        if fail_count >= patience:
            warnings.warn(f"Number of repeated samples exceeded patience ({patience} "
                          "tries)!", stacklevel=2)
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
    continue_sampling, fail_count = check_repeated_samples(trial_index_to_param, 
                                                           last_samples, fail_count)

    continue_optimization = energy_ok and continue_sampling
    return continue_optimization, fail_count
                

def setup_generation_strategy(num_random_trials=6):
    """
    Setup custom generation strategy.

    Parameters
    ----------
    num_random_trials : int, optional
        Number of random trials to perform before starting Bayesian optimization. 
        Default is 6.

    Returns
    -------
    ax.modelbridge.generation_strategy.GenerationStrategy
        The configured generation strategy.
    """
    gs = GenerationStrategy(
        nodes=[
            GenerationNode(
                name="sobol",
                generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
                transition_criteria=[
                    MinTrials(transition_to="gpei", threshold=num_random_trials)
                ],
            ),
            GenerationNode(
                name="gpei",
                generator_specs=[GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)],
            ),
        ]
    )
    return gs


def run_optimization(ax_client, executor, default_ini_file, n_trials, max_parallel_jobs,
                     experiment_file, metric_names=("N_total"), 
                     noiseless_observations=True):
    """
    Run Bayesian optimization using Ax and submitit.

    Parameters
    ----------
    ax_client : ax.api.client.Client
        Ax client for managing the optimization process.
    executor : submitit.AutoExecutor
        Submitit executor for running jobs on a cluster or locally.
    default_ini_file: string
        Path to the default initialization file.
    n_trials : int
        Total number of trials to run.
    max_parallel_jobs : int
        Maximum number of parallel jobs to run.
    experiment_file : str
        Path to save the Ax experiment data.
    metric_names : list of str, optional
        List of metric names to evaluate. Default is ("N_total").

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
            number_of_jobs_to_submit = min(
                max_parallel_jobs - len(jobs), n_trials - submitted_jobs
            )
            trial_index_to_param = {}
            if number_of_jobs_to_submit > 0:
                trial_index_to_param = ax_client.get_next_trials(
                    max_trials=number_of_jobs_to_submit
                )
            # Check that sampled trials satisfy the requirements
            if trial_index_to_param:
                flags = check_sampled_trials(trial_index_to_param,last_samples,
                                             fail_count)
                continue_optimization, fail_count = flags

                if continue_optimization:
                    for trial_idx, params in trial_index_to_param.items():
                        job = executor.submit(
                            quvac_evaluation, 
                            params, 
                            trial_idx,
                            default_ini_file,
                            metric_names,
                            noiseless_observations,
                        )
                        submitted_jobs += 1
                        jobs.append((job, trial_idx))
                        time.sleep(5)
                else:
                    warnings.warn("Terminating optimization... Finishing last "
                                  "trials...", stacklevel=2)
            else:
                time.sleep(5)
                    

def _create_new_ax_client(experiment_name,params_for_ax,parameter_constraints):
    ax_client = Client()
    ax_client.configure_experiment(
        parameters=params_for_ax,
        parameter_constraints=parameter_constraints,
        name=experiment_name,
    )
    _logger.info("Created a new experiment!")

    return ax_client


def setup_ax_client(
        experiment_file, 
        experiment_name,
        params_for_ax,
        parameter_constraints,
        start_fresh,
    ):
    """
    Create a new ax client or load it from the existing file.

    Parameters
    ----------
    experiment_file : str
        Path to save the Ax experiment data.
    experiment_name: str
        Experiment name.
    params_for_ax: list of ax.api.configs.RangeParameterConfig
        Parameters to optimize in ax-appropriate style.
    parameter_constraints: list of str
        Constraints for parameter space.
    start_fresh: bool
        Whether to remove the old experiment file if it would be found.

    Returns
    -------
    ax.api.client.Client
        Client with defined parameter landscape.
    """
    if start_fresh:
        if os.path.isfile(experiment_file):
            os.remove(experiment_file)
            _logger.info("Deleted old experiment...")
        ax_client = _create_new_ax_client(
            experiment_name,
            params_for_ax,
            parameter_constraints,
        )
    else:
        ax_client = Client.load_from_json_file(experiment_file)
        _logger.info("Loaded existing experiment!")
    
    return ax_client


def cluster_optimization(ini_file, save_path=None, wisdom_file=None):
    """
    Launch optimization of quvac simulation for a given initial configuration file.

    Parameters
    ----------
    ini_file : str
        Path to the initial configuration file (YAML format).
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
    files = get_dirs(ini_file, save_path, wisdom_file, mode="optimization")
    save_path = files['save_path']

    # Setup logger
    create_basic_logger(files["logger"])

    # Start time
    log_time(_logger, name="start")

    experiment_file = os.path.join(save_path, "experiment.json")

    ini_default = read_yaml(ini_file)
    optimization_params = ini_default["optimization"]
    cluster_params = optimization_params.get("cluster_params", {})

    # Check that optimization parameters are only field_parameters
    optimization_keys = list(optimization_params["parameters"].keys())
    err_msg = (
        f"Only field parameters could be optimized but you have {optimization_keys}"
    )
    assert sum([key.startswith("field") for key in optimization_keys]) == len(
        optimization_keys
    ), err_msg

    # Prepare parameters for optimization in ax style
    params_for_ax = prepare_params_for_ax(optimization_params["parameters"])

    # custom generation strategy
    number_of_ini_trials = optimization_params.get("number_of_ini_trials", 5)
    experiment_name = optimization_params.get("experiment_name", "test_optimization")

    # Set up optimization client
    parameter_constraints = optimization_params.get("parameter_constraints", None)
    start_fresh = optimization_params.get("start_fresh", True)

    ax_client = setup_ax_client(experiment_file, experiment_name, params_for_ax, 
                                parameter_constraints, start_fresh)

    objective = optimization_params["objective"]
    metrics_to_track = optimization_params.get("metrics_to_track", [])

    # Configure objective and tracking metrics
    ax_client.configure_optimization(
        objective=f"{objective}",
        outcome_constraints=optimization_params.get("outcome_constraints", None),
    )
    if metrics_to_track:
        ax_client.configure_tracking_metrics(metrics_to_track)

    gs_initialization_random_seed = optimization_params.get(
        "gs_initialization_random_seed", None
    )
    if gs_initialization_random_seed is None:
        gs_initialization_random_seed = int(np.random.randint(low=0, high=1000000))
    _logger.info(
        f"Initial random seed for generation strategy: {gs_initialization_random_seed}"
    )

    ax_client.configure_generation_strategy(
        method=optimization_params.get("generation_strategy_type", "fast"),
        initialization_budget=number_of_ini_trials,
    )

    max_parallel_jobs = cluster_params.get("max_parallel_jobs", 3)
    executor = setup_job_executor_from_params(cluster_params, save_path,
                                              max_parallel_jobs)
    metric_names = tuple([objective] + metrics_to_track)

    noiseless_observations = optimization_params.get("noiseless_observations", True)
    n_trials = optimization_params.get("n_trials", 10)

    _logger.info("MILESTONE: Optimization is fully configured.")
    _logger.info("Launching first trials...")

    run_optimization(ax_client, executor, ini_file, n_trials, max_parallel_jobs, 
                     experiment_file, metric_names, noiseless_observations)

    _logger.info("Optimization is finished!")

    # End time
    log_time(_logger, name="end")


def gather_trials_data(ax_client, metric_names=("N_total", "N_disc")):
    """
    Gather data from completed trials in the Ax experiment.

    Parameters
    ----------
    ax_client : ax.api.client.Client
        Ax client managing the optimization process.
    metric_names : list of str, optional
        List of metric names to extract from the trials. Default is 
        ("N_total", "N_disc").

    Returns
    -------
    dict
        Dictionary containing optimized parameters and metrics.
    """
    metrics = ax_client._experiment.fetch_data().df
    trials = ax_client._experiment.trials
    trials_params = {key: trial.arm.parameters for key, trial in trials.items()
                     if trial.completed_successfully}
    for key in trials_params:
        for metric_name in metric_names:
            condition = (metrics["metric_name"] == metric_name) & (
                metrics["trial_index"] == key
            )
            trials_params[key][metric_name] = metrics.loc[condition]["mean"].iloc[0]
    
    # join data from separate trials into arrays
    first_trial_key = list(trials_params.keys())[0]
    trial_keys = list(trials_params[first_trial_key].keys())
    trial_data = {}
    for key in trial_keys:
        trial_data[key] = np.array([val[key] for val in trials_params.values()])
    return trial_data


class SurrogateModel:
    """
    Restores surrogate model for a given optimization experiment.

    Parameters
    ----------
    ax_client: ax.api.client.Client
        Ax client managing the optimization process.
    metric: str, optional
        Metric of interest to collect. 
    """
    def __init__(self, ax_client, metric="N_total"):
        self.experiment_data = ax_client._experiment.fetch_data()
        self.model = Generators.BOTORCH_MODULAR(
            experiment=ax_client._experiment,
            data=self.experiment_data,
        )
        self.metric = metric

    def _update_input_parameters(self, model_input, fixed_params):
        if fixed_params:
            for point in model_input:
                point.update(fixed_params)
        model_input = [ObservationFeatures(parameters=pt) for pt in model_input]
        return model_input
    
    def _make_model_prediction(self, model_input, fixed_params):
        model_input = self._update_input_parameters(model_input, fixed_params)
        mean, covariance = self.model.predict(model_input)
        mean = np.array(mean[self.metric])
        covariance = np.array(covariance[self.metric][self.metric])
        return mean, covariance

    def predict_at_point(self, param_name, param_value, fixed_params=None):
        """
        Make a prediction for a particular parameter value.

        Parameters
        ----------
        param_name: str
            Parameter name for which the prediction would be made.
        param_value: int or float
            Parameter value (prediction point).
        fixed_params: dict of {param_name: param_value}, optional
            Fixed values of other parameters (useful for high-dimentional optimization).

        Returns
        -------
        (np.ndarray, np.ndarray)
            Mean and covariance values at a given parameter point.
        """
        model_input = [{param_name: param_value}]
        mean, covariance = self._make_model_prediction(model_input, fixed_params)
        return mean, covariance

    def predict_1d(self, param_name, param_values, fixed_params=None):
        """
        Make a prediction for a particular parameter value.

        Parameters
        ----------
        param_name: str
            Parameter name for which the prediction would be made.
        param_values: collections.abc.Iterable
            Array of parameter values.
        fixed_params: dict of {param_name: param_value}, optional
            Fixed values of other parameters (useful for high-dimentional optimization).

        Returns
        -------
        (np.ndarray, np.ndarray)
            Mean and covariance values for given array of parameter values.
        """
        err_msg = "Method `predict_1d` works only with sequences"
        assert isinstance(param_values, Iterable), err_msg
        model_input = [{param_name: value} for value in param_values]
        mean, covariance = self._make_model_prediction(model_input, fixed_params)
        return mean, covariance

    def predict_2d(self, param_names, param_values, fixed_params=None):
        """
        Make a prediction for a particular parameter value.

        Parameters
        ----------
        param_names: (str, str)
            Two parameter names for which the prediction would be made.
        param_values: (Iterable, Iterable)
            Two arrays of parameter values.
        fixed_params: dict of {param_name: param_value}, optional
            Fixed values of other parameters (useful for high-dimentional optimization).

        Returns
        -------
        (np.ndarray, np.ndarray)
            Mean and covariance values for given arrays of parameter values.
        """
        assert len(param_names) == 2, "Only two variable parameters are accepted"
        assert len(param_values) == 2, "Only two variable parameter grids are accepted"
        param_name_1, param_name_2 = param_names
        n1, n2 = len(param_values[0]), len(param_values[1])
        model_input = [
            {param_name_1: value1, param_name_2: value2}
            for value1,value2 in itertools.product(*param_values)
        ]
        mean, covariance = self._make_model_prediction(model_input, fixed_params)
        mean, covariance = mean.reshape((n1,n2)), covariance.reshape((n1,n2))
        return mean, covariance


def main_optimization():
    """
    Main function to run the optimization.
    """
    args = parse_args(description="Perform optimization of quvac simulations")
    cluster_optimization(args.input, args.output)


if __name__ == "__main__":
    main_optimization()
