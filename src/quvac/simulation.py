#!/usr/bin/env python3
"""
Script to launch Vacuum Emission simulation, do postprocessing
and measure performance.

Usage:

.. code-block:: bash

    python simulation.py -i <input>.yaml -o <output_dir> --wisdom <wisdom_file>
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

_logger = logging.getLogger("simulation")


def _parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    description = "Calculate quantum vacuum signal for given external fields."
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
    """
    Check if directories exist and create them if necessary.

    Parameters
    ----------
    ini_file : str
        Path to the initialization file.
    save_path : str
        Path to save the results.

    Returns
    -------
    save_path : str
        Validated save path.
    """
    assert os.path.isfile(ini_file), f"{ini_file} is not a file or does not exist"
    if save_path is None:
        save_path = os.path.dirname(ini_file)
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
    return save_path


def get_filenames(ini_file, save_path, wisdom_file):
    """
    Get filenames for saving simulation data.

    Parameters
    ----------
    ini_file : str
        Path to the initialization file.
    save_path : str
        Path to save the results.
    wisdom_file : str
        Path to save pyfftw-wisdom.

    Returns
    -------
    files : dict
        Dictionary containing filenames for saving simulation data.
    """
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
    """
    Get directories for saving simulation data.

    Parameters
    ----------
    ini_file : str
        Path to the initialization file.
    save_path : str
        Path to save the results.
    wisdom_file : str
        Path to save pyfftw-wisdom.

    Returns
    -------
    files : dict
        Dictionary containing directories for saving simulation data.
    """
    save_path = check_dirs(ini_file, save_path)
    files = get_filenames(ini_file, save_path, wisdom_file)
    return files


def set_precision(precision):
    """
    Set global precision for calculations.

    Parameters
    ----------
    precision : str
        Precision type, either "float32" or "float64".
    """
    if precision == "float32":
        config.FDTYPE = "float32"
        config.CDTYPE = "complex64"
    else:
        config.FDTYPE = "float64"
        config.CDTYPE = "complex128"


def run_simulation(ini_config, fields_params, files, timings, memory):
    """
    Run the vacuum emission simulation.

    Parameters
    ----------
    ini_config : dict
        Dictionary containing the initialization configuration.
    fields_params : list of dict
        List of dictionaries containing the field parameters.
    files : dict
        Dictionary containing filenames for saving simulation data.
    timings : dict
        Dictionary to store timing information.
    memory : dict
        Dictionary to store memory usage information.

    Returns
    -------
    timings : dict
        Updated dictionary with timing information.
    memory : dict
        Updated dictionary with memory usage information.
    """
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
    _logger.info(grid_print)
    _logger.info("MILESTONE: Grids are created\n")

    # Shorten time grid for the test run
    if test_run:
        expected_timesteps = len(grid_t)
        grid_t = grid_t[:test_timesteps]
        do_postprocess = False
        _logger.info(f"Performing test run for {test_timesteps} timesteps\n")

    # Field setup
    _logger.info(
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
    _logger.info("====================================================\n")
    _logger.info("MILESTONE: Fields are set up")

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
    _logger.info(log_message)
    vacem = VacuumEmission(field, grid_xyz, nthreads=pyfftw_threads, channels=channels)
    timings['vacem_setup'] = time.perf_counter()
    timings['integral'] = vacem.calculate_amplitudes(grid_t, save_path=files['amplitudes'])
    timings['amplitudes'] = time.perf_counter()
    memory['maxrss_amplitudes'] = get_maxrss()
    _logger.info("MILESTONE: Amplitudes are calculated")

    timings['per_iteration'] = (timings['amplitudes'] - timings['field_setup']) / len(grid_t)

    if test_run:
        test_run_str_print = get_test_timings(timings, len(grid_t), expected_timesteps)
        _logger.info(test_run_str_print)
        print(test_run_str_print)
    return timings, memory


def postprocess_simulation(ini_config, files, fields_params):
    """
    Perform postprocessing on the simulation data.

    Parameters
    ----------
    ini_config : dict
        Dictionary containing the initialization configuration.
    files : dict
        Dictionary containing filenames for saving simulation data.
    fields_params : list of dict
        List of dictionaries containing the field parameters.
    """
    # Get postprocess params from ini config
    postprocess_params = ini_config.get("postprocess", {})

    kwargs = {
        "perp_type": postprocess_params.get("perp_polarization_type", None),
        "perp_field_idx": postprocess_params.get("perp_field_idx", 1),
        "stokes": postprocess_params.get("stokes", False),
        "calculate_xyz_background": postprocess_params.get("calculate_xyz_background", False),
        "bgr_idx": postprocess_params.get("bgr_idx", False),
        "calculate_spherical": postprocess_params.get("calculate_spherical", False),
        "spherical_params": postprocess_params.get("spherical_params", {}),
        "calculate_discernible": postprocess_params.get("calculate_discernible", False),
        "discernibility": postprocess_params.get("discernibility", "angular"),
        "add_signal_bg": postprocess_params.get("add_signal_bg", False),
    }

    # Do postprocessing
    postprocess_print = get_postprocess_info(postprocess_params)
    _logger.info(postprocess_print)

    modes = postprocess_params.get("modes", ["total", "polarization"])
    if kwargs["perp_type"] is None:
         modes = ["total"]
    for mode in modes:
        save_path = files['spectra'].replace(".npz", f"_{mode}.npz")
        analyzer = VacuumEmissionAnalyzer(
            fields_params, data_path=files['amplitudes'], save_path=save_path
        )
        analyzer.get_spectra(
            mode=mode,
            **kwargs
        )
    _logger.info("MILESTONE: Spectra are calculated from amplitudes")


def quvac_simulation(ini_file, save_path=None, wisdom_file="wisdom/fftw-wisdom"):
    """
    Launch a single quvac simulation for given <ini>.yaml file.

    Parameters
    ----------
    ini_file : str
        Path to the initialization file.
    save_path : str, optional
        Path to save simulation results to, by default None.
    wisdom_file : str, optional
        Path to save pyfftw-wisdom, by default "wisdom/fftw-wisdom".
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
    _logger.info(start_print)
    timings = {}
    timings['start'] = time.perf_counter()
    memory = {'maxrss_amplitudes': 0}

    # Set up global precision for calculations
    perf_params = ini_config.get("performance", {})
    precision = perf_params.get("precision", "float64")
    set_precision(precision)
    _logger.info(f"Using {precision} precision")

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
        _logger.info(perf_print)
    elif do_postprocess:
        perf_print = get_postprocess_stats(perf_stats)
        print(perf_print)
        _logger.info(perf_print)

    print("Simulation finished!")

    # End time
    time_log_end = time.asctime(time.localtime())
    end_print = simulation_end_str.format(time_log_end)
    _logger.info(end_print)


if __name__ == "__main__":
    args = _parse_args()
    quvac_simulation(args.input, args.output, args.wisdom)
