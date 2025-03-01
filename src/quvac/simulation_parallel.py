#!/usr/bin/env python3
"""
Script to launch Vacuum Emission simulation IN PARALLEL 
(using submitit for semi-slurm submission type), do postprocessing
and measure performance.

Usage:

.. code-block:: bash

    python simulation_parallel.py -i <input>.yaml -o <output_dir> 
    --wisdom <wisdom_file>
"""
import argparse
import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import submitit

from quvac.config import DEFAULT_SUBMITIT_PARAMS
from quvac.grid import setup_grids
from quvac.log import (get_parallel_performance_stats,
                       simulation_end_str, simulation_start_str)
from quvac.simulation import get_dirs, quvac_simulation, postprocess_simulation
from quvac.utils import read_yaml, write_yaml, get_maxrss

_logger = logging.getLogger("simulation")


def create_ini_files_for_parallel(ini_config, grid_xyz, grid_t, n_jobs, save_path):
    """
    Create initialization files for parallel jobs.

    Parameters
    ----------
    ini_config : dict
        Dictionary containing the initialization configuration.
    grid_xyz : quvac.grid.GridXYZ
        The spatial grid object.
    grid_t : numpy.ndarray
        The temporal grid array.
    n_jobs : int
        Number of parallel jobs.
    save_path : str
        Path to save the initialization files.

    Returns
    -------
    ini_files : list of str
        List of paths to the initialization files for each job.
    """
    Nt_total = len(grid_t)
    Nt_per_job = Nt_total // n_jobs

    ini_job = deepcopy(ini_config)
    ini_job["mode"] = "simulation"
    ini_job["postprocess"] = {}

    box_xyz = [float(-ax[0] * 2) for ax in grid_xyz.grid]
    Nxyz = [int(N) for N in grid_xyz.grid_shape]
    grid_params = {"mode": "direct", "box_xyz": box_xyz, "Nxyz": Nxyz}
    ini_job["grid"].update(grid_params)
    # Create ini files for all jobs
    ini_files = []
    for idx in range(n_jobs):
        idx_start, idx_end = Nt_per_job * idx, Nt_per_job * (idx + 1) - 1
        Nt = Nt_per_job
        if (n_jobs - 1) == idx and n_jobs > 1:
            idx_end += 1
            Nt += 1
        box_t = [float(grid_t[idx_start]), float(grid_t[idx_end])]
        grid_params_job = {
            "box_t": box_t,
            "Nt": Nt,
        }
        ini_job["grid"].update(grid_params_job)
        ini_path_job = os.path.join(save_path, f"job_{str(idx).zfill(2)}", "ini.yml")
        Path(os.path.dirname(ini_path_job)).mkdir(parents=True, exist_ok=True)
        write_yaml(ini_path_job, ini_job)
        ini_files.append(ini_path_job)
    return ini_files


def collect_results(ini_files, amplitudes_file):
    """
    Collect results from parallel jobs and combine them.

    Parameters
    ----------
    ini_files : list of str
        List of paths to the initialization files for each job.
    amplitudes_file : str
        Path to save the combined amplitudes.
    """
    amplitude_files = [
        os.path.join(os.path.dirname(ini_file), "amplitudes.npz")
        for ini_file in ini_files
    ]
    amplitude_0 = np.load(amplitude_files[0])
    x, y, z = [amplitude_0[ax] for ax in "xyz"]
    S1, S2 = 0, 0
    for amplitude in amplitude_files:
        data = np.load(amplitude)
        S1 += data["S1"]
        S2 += data["S2"]
    amplitude_total = {"x": x, "y": y, "z": z, "S1": S1, "S2": S2}
    np.savez(amplitudes_file, **amplitude_total)


def _parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    description = "Calculate quantum vacuum signal for given external fields"
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


def quvac_simulation_parallel(
    ini_file, save_path=None, wisdom_file="wisdom/fftw-wisdom"
):
    """
    Launch a single quvac simulation for given <ini>.yaml file in parallel.

    Depending on available jobs, we split the total time interval
    into several sub-intervals and submit each sub-interval for
    calculation as a separate quvac simulation (without postprocessing).
    Then we gather all S1 and S2 in the main process and do main postprocessing.

    Parameters
    ----------
    ini_file : str
        Path to the initialization file.
    save_path : str, optional
        Path to save simulation results to, by default None.
    wisdom_file : str, optional
        Path to save pyfftw-wisdom, by default "wisdom/fftw-wisdom".
    """
    # Check that ini file and save_path exists
    files = get_dirs(ini_file, save_path, wisdom_file)

    # Load and parse ini yaml file
    ini_config = read_yaml(ini_file)
    # One can choose either to perform just simulation, just postprocess or both
    mode = ini_config.get('mode', 'simulation_postprocess')
    do_simulation = 'simulation' in mode
    do_postprocess = 'postprocess' in mode

    # Setup logger
    logging.basicConfig(
        filename=files['logger'],
        filemode="w",
        encoding="utf-8",
        level=logging.DEBUG,
        format=f"%(message)s",
    )

    # Start time
    time_log_start = time.asctime(time.localtime())
    start_print = simulation_start_str.format(time_log_start)
    _logger.info(start_print)
    timings = {}
    timings['start'] = time.perf_counter()

    # Load and parse ini yaml file
    fields_params = ini_config["fields"]
    if isinstance(fields_params, dict):
        fields_params = list(fields_params.values())
    grid_params = ini_config["grid"]
    cluster_params = ini_config.get("cluster_params", {})

    # Get parallelization params
    n_jobs = cluster_params.get("n_jobs", 2)
    max_jobs = cluster_params.get("max_jobs", n_jobs)
    cluster = cluster_params.get("cluster", "local")
    sbatch_params = cluster_params.get("sbatch_params", DEFAULT_SUBMITIT_PARAMS)

    # Get grids
    grid_xyz, grid_t = setup_grids(fields_params, grid_params)

    ini_files = create_ini_files_for_parallel(
        ini_config, grid_xyz, grid_t, n_jobs, files['save_path']
    )

    # Create a cluster
    submitit_folder = os.path.join(files['save_path'], "submitit_logs")
    executor = submitit.AutoExecutor(folder=submitit_folder, cluster=cluster)
    if cluster == "slurm":
        executor.update_parameters(slurm_array_parallelism=max_jobs)
        executor.update_parameters(**sbatch_params)

    # Submit jobs
    _logger.info("MILESTONE: Submitting jobs...")
    jobs = executor.map_array(quvac_simulation, ini_files)
    _logger.info("MILESTONE: Jobs submitted, waiting for results...")

    # Wait till all jobs end
    outputs = [job.result() for job in jobs]
    _logger.info("MILESTONE: Jobs are finished")

    # Collect all results
    collect_results(ini_files, files['amplitudes'])
    _logger.info("MILESTONE: Results from individual jobs are collected")
    timings['jobs'] = time.perf_counter()

    maxrss_jobs = get_maxrss()

    # Calculate spectra
    if do_postprocess:
        postprocess_simulation(ini_config, files, fields_params)

    timings['postprocess'] = time.perf_counter()
    timings['total'] = timings['postprocess'] - timings['start']

    maxrss_total = get_maxrss()

    memory = {"maxrss_jobs": maxrss_jobs, "maxrss_total": maxrss_total}
    perf_stats = {"timings": timings, "memory": memory}

    write_yaml(files['performance'], perf_stats)

    perf_print = get_parallel_performance_stats(perf_stats)
    print(perf_print)
    _logger.info(perf_print)

    print("Simulation finished!")

    # End time
    time_log_end = time.asctime(time.localtime())
    end_print = simulation_end_str.format(time_log_end)
    _logger.info(end_print)


if __name__ == "__main__":
    args = _parse_args()
    quvac_simulation_parallel(args.input, args.output, args.wisdom)
