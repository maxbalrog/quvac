"""
Utilities to run simulations in parallel either locally or in Slurm environment.
"""
import logging
import os

import submitit

from quvac.config import DEFAULT_SLURM_PARAMS
from quvac.simulation import quvac_simulation
from quvac.utils import estimate_memory_usage

_logger = logging.getLogger("simulation")


def slurm_time_to_mins(slurm_time):
    """
    Converts Slurm time in format `DAYS-HOURS:MINUTES:SECONDS` to minutes.

    Parameters
    ----------
    slurm_time: str
        Time formatted in Slurm format.

    Returns
    -------
    int
        Time in minutes.
    """
    days = 0
    if "-" in slurm_time:
        days, slurm_time = slurm_time.split("-")
    hours, mins, seconds =  slurm_time.split(":")
    days, hours, mins = [int(t) for t in (days, hours, mins)]
    return days*24*60 + hours*60 + mins


def slurm_to_submitit_params(sbatch_params):
    """
    Transform Slurm keywords to relevant ones for submitit.

    Parameters
    ----------
    sbatch_params: dict
        Parameters for sbatch script.

    Returns
    -------
    dict
        Parameters for sbatch script in submitit format.
    """
    keys_to_copy = {"slurm_partition": "cpu", "cpus_per_task": 4}
    submitit_params = {
        key: sbatch_params.get(key, default_val) 
        for key,default_val in keys_to_copy.items()
    }
    submitit_params["slurm_mem"] = sbatch_params["memory"]
    submitit_params["timeout_min"] = slurm_time_to_mins(sbatch_params["walltime"])
    return submitit_params


def create_job_executor(cluster_type, executor_folder, sbatch_params,
                        max_parallel_jobs):
    """
    Create submitit job executor.

    Parameters
    ----------
    cluster_type: 'local' or 'slurm'
        On which machine the calculation would be performed.
    executor_folder: str
        Path where to create a folder for executor files and logs.
    sbatch_params: dict
        Parameters of individual Slurm job.
    max_parallel_jobs: int
        Maximal number of jobs to run concurrently.

    Returns
    -------
    submitit.AutoExecutor
        Executor for running jobs.
    """
    executor = submitit.AutoExecutor(folder=executor_folder, cluster=cluster_type)
    if cluster_type == "slurm":
        executor.update_parameters(slurm_array_parallelism=max_parallel_jobs)
    executor.update_parameters(**sbatch_params)
    return executor


def setup_job_executor_from_params(cluster_params, save_path, 
                                   max_parallel_jobs_default=2):
    """
    Setup job executor from unified cluster parameters.

    Parameters
    ----------
    cluster_params: dict
        Parameters of parallel execution.
    save_path : str
        Path where to create a folder for executor files and logs.
    max_parallel_jobs_default: int, optional
        Default maximal number of jobs to run concurrently.

    Returns
    -------
    submitit.AutoExecutor
        Executor for running jobs.
    """
    max_parallel_jobs = cluster_params.get("max_parallel_jobs", 
                                           max_parallel_jobs_default)
    cluster_type = cluster_params.get("cluster_type", "local")
    sbatch_params = cluster_params.get("sbatch_params", DEFAULT_SLURM_PARAMS)
    sbatch_params = slurm_to_submitit_params(sbatch_params)

    # Create submitit executor
    submitit_folder = os.path.join(save_path, "submitit_files")
    executor = create_job_executor(cluster_type, submitit_folder, sbatch_params,
                                   max_parallel_jobs)
    return executor


def submit_jobs_with_memory_estimation(executor, ini_files):
    """
    Submit jobs for a list of initialization files estimating memory usage
    for each of them.

    Parameters
    ----------
    executor : submitit.AutoExecutor
        Executor for running jobs.
    ini_files : list of str
        List of paths to the initialization files for each job.

    Returns
    -------
    list of submitit jobs
        Submitted jobs.
    """
    jobs = []
    for ini_file in ini_files:
        memory = estimate_memory_usage(ini_file)
        executor.update_parameters(slurm_mem=memory)
        job = executor.submit(quvac_simulation, ini_file)
        jobs.append(job)
    return jobs


def run_simulations_with_job_executor(ini_files, cluster_params, save_path,
                                      max_parallel_jobs_default=2,):
    """
    Run simulations for a list of ini files with job executor.

    Parameters
    ----------
    ini_files : list of str
        List of paths to the initialization files for each job.
    cluster_params: dict
        Parameters of parallel execution.
    save_path : str
        Path where to create a folder for submitit files and logs.
    max_parallel_jobs_default: int, optional
        Maximal number of jobs to run concurrently.
    """
    executor = setup_job_executor_from_params(cluster_params, save_path, 
                                              max_parallel_jobs_default)

    # Submit jobs
    estimate_memory_usage = cluster_params.get("estimate_memory_usage", False)
    _logger.info("MILESTONE: Submitting jobs...")
    if estimate_memory_usage:
        jobs = submit_jobs_with_memory_estimation(executor, ini_files)
    else:
        jobs = executor.map_array(quvac_simulation, ini_files)
    _logger.info("MILESTONE: Jobs submitted, waiting for results...")

    # Wait till all jobs end
    _ = [job.result() for job in jobs]
    _logger.info("MILESTONE: Jobs are finished")

