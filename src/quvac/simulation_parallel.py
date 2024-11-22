#!/usr/bin/env python3
'''
Here we provide a script to launch Vacuum Emission simulation
IN PARALLEL (using submitit for semi-slurm submission type),
do postprocessing and measure performance
'''
import argparse
import logging
import os
from pathlib import Path
import time
import resource
from copy import deepcopy

import numpy as np
import submitit

from quvac.config import DEFAULT_SUBMITIT_PARAMS
from quvac.grid import setup_grids
from quvac.log import (simulation_start_str, simulation_end_str,
                       get_grid_params, get_parallel_performance_stats,
                       get_postprocess_info, test_run_str)
from quvac.simulation import quvac_simulation
from quvac.postprocess import VacuumEmissionAnalyzer
from quvac.utils import read_yaml, write_yaml


logger = logging.getLogger('simulation')


def create_ini_files_for_parallel(ini_config, grid_xyz, grid_t, n_jobs,
                                  save_path):
    Nt_total = len(grid_t)
    Nt_per_job = Nt_total // n_jobs

    ini_job = deepcopy(ini_config)
    ini_job['postprocess'] = {}

    box_xyz = [float(-ax[0]*2) for ax in grid_xyz.grid]
    Nxyz = [int(N) for N in grid_xyz.grid_shape]
    grid_params = {
        'mode': 'direct',
        'box_xyz': box_xyz,
        'Nxyz': Nxyz
    }
    ini_job['grid'].update(grid_params)
    # Create ini files for all jobs
    ini_files = []
    for idx in range(n_jobs):
        idx_start, idx_end = Nt_per_job*idx, Nt_per_job*(idx+1)-1
        Nt = Nt_per_job
        if (n_jobs-1) == idx and n_jobs > 1:
            idx_end += 1
            Nt += 1
        box_t = [float(grid_t[idx_start]), float(grid_t[idx_end])]
        grid_params_job = {
            'box_t': box_t,
            'Nt': Nt,
        }
        ini_job['grid'].update(grid_params_job)
        ini_path_job = os.path.join(save_path, f'job_{idx}', 'ini.yml')
        Path(os.path.dirname(ini_path_job)).mkdir(parents=True, exist_ok=True)
        write_yaml(ini_path_job, ini_job)
        ini_files.append(ini_path_job)
    return ini_files


def collect_results(ini_files, amplitudes_file):
    amplitude_files = [os.path.join(os.path.dirname(ini_file), 'amplitudes.npz')
                       for ini_file in ini_files]
    amplitude_0 = np.load(amplitude_files[0])
    x, y, z = [amplitude_0[ax] for ax in 'xyz']
    S1, S2 = 0, 0
    for amplitude in amplitude_files:
        data = np.load(amplitude)
        S1 += data['S1']
        S2 += data['S2']
    amplitude_total = {
        'x': x,
        'y': y,
        'z': z,
        'S1': S1,
        'S2': S2
    }
    np.savez(amplitudes_file, **amplitude_total)


def parse_args():
    description = "Calculate quantum vacuum signal for given external fields"
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument("--input", "-i", default=None,
                           help="Input yaml file with field and grid params")
    argparser.add_argument("--output", "-o", default=None,
                           help="Path to save simulation data to")
    argparser.add_argument("--wisdom", default='wisdom/fftw-wisdom',
                           help="File to save pyfftw-wisdom")
    return argparser.parse_args()


def quvac_simulation_parallel(ini_file, save_path=None, wisdom_file='wisdom/fftw-wisdom'):
    '''
    We read ini script and parallelization parameters
    Depending on available jobs, we split the total time interval
    into several sub-intervals and submit each sub-interval for 
    calculation as a separate quvac simulation (without postprocessing).
    Then we gather all S1 and S2 in the main process and do main postprocessing.
    '''
     # Check that ini file and save_path exists
    assert os.path.isfile(ini_file), f"{ini_file} is not a file or does not exist"
    if save_path is None:
        save_path = os.path.dirname(ini_file)
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
    amplitudes_file = os.path.join(save_path, 'amplitudes.npz')
    spectra_file = os.path.join(save_path, 'spectra.npz')
    performance_file = os.path.join(save_path, 'performance.yml')
    
    # Setup logger
    logger_file = os.path.join(save_path, 'simulation.log')
    logging.basicConfig(filename=logger_file, filemode='w', encoding='utf-8',
                        level=logging.DEBUG, format=f'%(message)s')
    
    # Start time
    time_log_start = time.asctime(time.localtime())
    start_print = simulation_start_str.format(time_log_start)
    logger.info(start_print)

    # Load and parse ini yaml file
    ini_config = read_yaml(ini_file)
    fields_params = ini_config["fields"]
    if isinstance(fields_params, dict):
        fields_params = list(fields_params.values())
    grid_params = ini_config["grid"]
    cluster_params = ini_config.get("cluster_params", {})

    # Determine postprocess steps
    postprocess_params = ini_config.get('postprocess', {})
    do_postprocess = True if postprocess_params else False
    if do_postprocess:
        calculate_spherical = postprocess_params.get('calculate_spherical', False)
        spherical_params = postprocess_params.get('spherical_params', {})
        calculate_discernible = postprocess_params.get('calculate_discernible', False)
        perp_type = postprocess_params.get('perp_polarization_type', None)
        perp_field_idx = postprocess_params.get('perp_field_idx', 1)

    # Get parallelization params
    n_jobs = cluster_params.get('n_jobs', 2)
    max_jobs = cluster_params.get('max_jobs', n_jobs)
    cluster = cluster_params.get('cluster', 'local')
    sbatch_params = cluster_params.get('sbatch_params', DEFAULT_SUBMITIT_PARAMS)

    # Get grids
    grid_xyz, grid_t = setup_grids(fields_params, grid_params)

    ini_files = create_ini_files_for_parallel(ini_config, grid_xyz, grid_t,
                                              n_jobs, save_path)
    

    time_start = time.perf_counter()
    # Create a cluster
    submitit_folder = os.path.join(save_path, 'submitit_logs')
    executor = submitit.AutoExecutor(folder=submitit_folder, cluster=cluster)
    if cluster == 'slurm':
        executor.update_parameters(slurm_array_parallelism=max_jobs)
        executor.update_parameters(**sbatch_params)

    # Submit jobs
    logger.info('MILESTONE: Submitting jobs...')
    jobs = executor.map_array(quvac_simulation, ini_files)
    logger.info('MILESTONE: Jobs submitted, waiting for results...')

    # Wait till all jobs end
    outputs = [job.result() for job in jobs]
    logger.info('MILESTONE: Jobs are finished')

    # Collect all results
    collect_results(ini_files, amplitudes_file)
    logger.info('MILESTONE: Results from individual jobs are collected')
    time_jobs_finished = time.perf_counter()

    maxrss_jobs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Calculate spectra
    if do_postprocess:
        postprocess_print = get_postprocess_info(postprocess_params)
        logger.info(postprocess_print)
        analyzer = VacuumEmissionAnalyzer(fields_params, data_path=amplitudes_file,
                                        save_path=spectra_file)
        analyzer.get_spectra(perp_field_idx=perp_field_idx,
                            perp_type=perp_type,
                            calculate_spherical=calculate_spherical,
                            spherical_params=spherical_params,
                            calculate_discernible=calculate_discernible)
        logger.info('MILESTONE: Spectra are calculated from amplitudes')
    time_postprocess = time.perf_counter()

    timings = {
        'start': time_start,
        'jobs': time_jobs_finished,
        'postprocess': time_postprocess,
        'total': time_postprocess-time_start,
    }

    maxrss_total = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    memory = {
        'maxrss_jobs': maxrss_jobs,
        'maxrss_total': maxrss_total
    }

    perf_stats = {
        'timings': timings,
        'memory': memory
    }

    write_yaml(performance_file, perf_stats)

    perf_print = get_parallel_performance_stats(perf_stats)
    print(perf_print)
    logger.info(perf_print)

    print("Simulation finished!")

    # End time
    time_log_end = time.asctime(time.localtime())
    end_print = simulation_end_str.format(time_log_end)
    logger.info(end_print)


if __name__ == '__main__':
    args = parse_args()
    quvac_simulation_parallel(args.input, args.output, args.wisdom)


