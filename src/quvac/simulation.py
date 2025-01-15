#!/usr/bin/env python3
"""
Here we provide a script to launch Vacuum Emission simulation,
do postprocessing and measure performance
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numexpr as ne
import pyfftw

from quvac import config
from quvac.field.external_field import ExternalField, ProbePumpField
from quvac.grid import setup_grids
from quvac.integrator.vacuum_emission import VacuumEmission
from quvac.log import (get_grid_params, get_performance_stats,
                       get_postprocess_info, simulation_end_str,
                       simulation_start_str, get_test_timings,
                       get_postprocess_stats)
from quvac.postprocess import VacuumEmissionAnalyzer
from quvac.utils import (load_wisdom, read_yaml, save_wisdom,
                         write_yaml, get_maxrss)

logger = logging.getLogger("simulation")

# ini yaml structure
"""
fields:
    field_1:
        ...
    field_2:
        ...
    ...
grids (one of two modes):
    mode: 'direct'
    box_xyz: (xbox, ybox, zbox)
    Nxyz: (Nx, Ny, Nz)
    box_t: tbox
    Nt: Nt

    mode: 'dynamic'
    box_xyz: {'longitudinal': ..., 'transverse': ...}
    res_xyz: {'longitudinal': ..., 'transverse': ...}
    box_t: ...
    res_t: ...
    In 'dynamic' mode ... should be replaced by appropriate factors
    determining how grid size and resolution would be upscaled
    E.g., longitudinal xyz box would be upscaled with c*tau, trasverse - with w0
performance:
    nthreads: ...
"""


def parse_args():
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


def check_dirs(ini_file, save_path):
    assert os.path.isfile(ini_file), f"{ini_file} is not a file or does not exist"
    if save_path is None:
        save_path = os.path.dirname(ini_file)
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
    return save_path


def get_filenames(ini_file, save_path, wisdom_file):
    ini_config = read_yaml(ini_file)
    mode = ini_config.get('mode', 'simulation_postprocess')
    files = {}
    files['save_path'] = save_path
    files['ini'] = ini_file
    files['wisdom'] = wisdom_file
    files['amplitudes'] = os.path.join(save_path, "amplitudes.npz")
    files['spectra'] = os.path.join(save_path, "spectra.npz")
    files['performance'] = os.path.join(save_path, f"{mode}_performance.yml")
    files['logger'] = os.path.join(save_path, f"{mode}.log")
    return files


def get_dirs(ini_file, save_path, wisdom_file):
    save_path = check_dirs(ini_file, save_path)
    files = get_filenames(ini_file, save_path, wisdom_file)
    return files


def set_precision(precision):
    if precision == "float32":
        config.FDTYPE = "float32"
        config.CDTYPE = "complex64"
    else:
        config.FDTYPE = "float64"
        config.CDTYPE = "complex128"


def run_simulation(ini_config, fields_params, files, timings, memory):
    grid_params = ini_config["grid"]
    perf_params = ini_config.get("performance", {})

    # Determine integrator type
    integrator_params = ini_config.get("integrator", {})
    integrator_type = integrator_params.get("type", "vacuum_emission")
    channels = integrator_type.endswith("channels")
    if channels:
        probe_pump_idx = integrator_params.get("probe_pump_idx", None)

    # Set up number of threads
    nthreads = perf_params.get("nthreads", os.cpu_count())
    ne.set_num_threads(nthreads)
    pyfftw_threads = perf_params.get("pyfftw_threads", nthreads)
    use_wisdom = perf_params.get("use_wisdom", True)

    perf_params = ini_config.get("performance", {})

    # Check if it's a test run to plan resources
    test_run = perf_params.get("test_run", False)
    test_timesteps = perf_params.get("test_timesteps", 5)

    # Load fftw-wisdom if possible
    if use_wisdom and os.path.exists(files['wisdom']):
        pyfftw.import_wisdom(load_wisdom(files['wisdom']))

    # Get grids
    grid_xyz, grid_t = setup_grids(fields_params, grid_params)
    grid_xyz.get_k_grid()
    grid_print = get_grid_params(grid_xyz, grid_t)
    logger.info(grid_print)
    logger.info("MILESTONE: Grids are created\n")

    # Shorten time grid for the test run
    if test_run:
        expected_timesteps = len(grid_t)
        grid_t = grid_t[:test_timesteps]
        do_postprocess = False
        logger.info(f"Performing test run for {test_timesteps} timesteps\n")

    # Field setup
    logger.info(
        "Field constructor:\n" "===================================================="
    )
    if not channels:
        field = ExternalField(fields_params, grid_xyz, nthreads=pyfftw_threads)
    else:
        field = ProbePumpField(
            fields_params,
            grid_xyz,
            probe_pump_idx=probe_pump_idx,
            nthreads=pyfftw_threads,
        )
    timings['field_setup'] = time.perf_counter()
    logger.info("====================================================\n")
    logger.info("MILESTONE: Fields are set up")

    # Calculate amplitudes
    if not channels:
        log_message = "Calculating vacuum emission amplitude with external field..."
    else:
        probe_pump = field.probe_pump_idx
        log_message = (
            "Calculating vacuum emission amplitude for probe channel...\n"
            f"    Probe idx: {probe_pump['probe']}\n"
            f"    Pump  idx: {probe_pump['pump']}"
        )
    logger.info(log_message)
    vacem = VacuumEmission(field, grid_xyz, nthreads=pyfftw_threads, channels=channels)
    timings['vacem_setup'] = time.perf_counter()
    timings['integral'] = vacem.calculate_amplitudes(grid_t, save_path=files['amplitudes'])
    timings['amplitudes'] = time.perf_counter()
    memory['maxrss_amplitudes'] = get_maxrss()
    logger.info("MILESTONE: Amplitudes are calculated")

    timings['per_iteration'] = (timings['amplitudes'] - timings['field_setup']) / len(grid_t)

    if test_run:
        test_run_str_print = get_test_timings(timings, len(grid_t), expected_timesteps)
        logger.info(test_run_str_print)
        print(test_run_str_print)
    return timings, memory


def postprocess_simulation(ini_config, files, fields_params):
    # Get postprocess params from ini config
    postprocess_params = ini_config.get("postprocess", {})

    kwargs = {
        "perp_type": postprocess_params.get("perp_polarization_type", None),
        "perp_field_idx": postprocess_params.get("perp_field_idx", 1),
        "calculate_xyz_background": postprocess_params.get("calculate_xyz_background", False),
        "calculate_spherical": postprocess_params.get("calculate_spherical", False),
        "spherical_params": postprocess_params.get("spherical_params", {}),
        "calculate_discernible": postprocess_params.get("calculate_discernible", False),
        "discernibility": postprocess_params.get("discernibility", "angular"),
    }

    # Do postprocessing
    postprocess_print = get_postprocess_info(postprocess_params)
    logger.info(postprocess_print)

    modes = postprocess_params.get("modes", "total polarization").split()
    for mode in modes:
        save_path = files['spectra'].replace(".npz", f"_{mode}.npz")
        analyzer = VacuumEmissionAnalyzer(
            fields_params, data_path=files['amplitudes'], save_path=save_path
        )
        analyzer.get_spectra(
            mode=mode,
            **kwargs
        )
    logger.info("MILESTONE: Spectra are calculated from amplitudes")


def quvac_simulation(ini_file, save_path=None, wisdom_file="wisdom/fftw-wisdom"):
    """
    Launch a single quvac simulation for given <ini>.yaml file

    Parameters:
    -----------
    ini_file: str (format <path>/<file_name>.yaml)
        Initial configuration file containing all simulation parameters
    save_path: str
        Path to save simulation results to
    """
    # Check that ini file and save_path exist
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
    logger.info(start_print)
    timings = {}
    timings['start'] = time.perf_counter()
    memory = {'maxrss_amplitudes': 0}

    # Set up global precision for calculations
    perf_params = ini_config.get("performance", {})
    precision = perf_params.get("precision", "float64")
    set_precision(precision)
    logger.info(f"Using {precision} precision")

    fields_params = ini_config["fields"]
    if isinstance(fields_params, dict):
        fields_params = list(fields_params.values())

    if do_simulation:
        timings, memory = run_simulation(ini_config, fields_params, 
                                         files, timings, memory)
    # Calculate spectra
    if do_postprocess:
        postprocess_simulation(ini_config, files, fields_params)
    timings['postprocess'] = time.perf_counter()
    timings['total'] = timings['postprocess'] - timings['start']

    # Save gained wisdom (for fftw)
    save_wisdom(ini_file, wisdom_file)

    memory['maxrss_total'] = get_maxrss()
    perf_stats = {"timings": timings, "memory": memory}

    write_yaml(files['performance'], perf_stats)

    if do_simulation:
        perf_print = get_performance_stats(perf_stats)
        print(perf_print)
        logger.info(perf_print)
    elif do_postprocess:
        perf_print = get_postprocess_stats(perf_stats)
        print(perf_print)
        logger.info(perf_print)

    print("Simulation finished!")

    # End time
    time_log_end = time.asctime(time.localtime())
    end_print = simulation_end_str.format(time_log_end)
    logger.info(end_print)


if __name__ == "__main__":
    args = parse_args()
    quvac_simulation(args.input, args.output, args.wisdom)
