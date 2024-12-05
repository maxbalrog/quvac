#!/usr/bin/env python3
"""
Here we provide a script to launch Vacuum Emission simulation,
do postprocessing and measure performance
"""

import argparse
import logging
import os
from pathlib import Path
import time
import resource

import numexpr as ne
import pyfftw

from quvac import config
from quvac.log import (
    simulation_start_str,
    simulation_end_str,
    get_grid_params,
    get_performance_stats,
    get_postprocess_info,
    test_run_str,
)
from quvac.field.external_field import ExternalField, ProbePumpField
from quvac.integrator.vacuum_emission import VacuumEmission
from quvac.grid import setup_grids
from quvac.postprocess import VacuumEmissionAnalyzer
from quvac.utils import read_yaml, write_yaml, load_wisdom, save_wisdom, format_time


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
    # Check that ini file and save_path exists
    assert os.path.isfile(ini_file), f"{ini_file} is not a file or does not exist"
    if save_path is None:
        save_path = os.path.dirname(ini_file)
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
    amplitudes_file = os.path.join(save_path, "amplitudes.npz")
    spectra_file = os.path.join(save_path, "spectra.npz")
    performance_file = os.path.join(save_path, "performance.yml")

    # Setup logger
    logger_file = os.path.join(save_path, "simulation.log")
    logging.basicConfig(
        filename=logger_file,
        filemode="w",
        encoding="utf-8",
        level=logging.DEBUG,
        format=f"%(message)s",
    )

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
    perf_params = ini_config.get("performance", {})

    # Determine integrator type
    integrator_params = ini_config.get("integrator", {})
    integrator_type = integrator_params.get("type", "vacuum_emission")
    channels = integrator_type.endswith("channels")
    if channels:
        probe_pump_idx = integrator_params.get("probe_pump_idx", None)

    # Determine postprocess steps
    postprocess_params = ini_config.get("postprocess", {})
    do_postprocess = True if postprocess_params else False
    if do_postprocess:
        calculate_spherical = postprocess_params.get("calculate_spherical", False)
        spherical_params = postprocess_params.get("spherical_params", {})
        calculate_discernible = postprocess_params.get("calculate_discernible", False)
        perp_type = postprocess_params.get("perp_polarization_type", None)
        perp_field_idx = postprocess_params.get("perp_field_idx", 1)

    # Set up number of threads
    nthreads = perf_params.get("nthreads", os.cpu_count())
    ne.set_num_threads(nthreads)
    pyfftw_threads = perf_params.get("pyfftw_threads", nthreads)
    # Set up global precision for calculations
    precision = perf_params.get("precision", "float64")
    if precision == "float32":
        config.FDTYPE = "float32"
        config.CDTYPE = "complex64"
    else:
        config.FDTYPE = "float64"
        config.CDTYPE = "complex128"
    use_wisdom = perf_params.get("use_wisdom", True)
    logger.info(f"Using {precision} precision")

    # Check if it's a test run to plan resources
    test_run = perf_params.get("test_run", False)
    test_timesteps = perf_params.get("test_timesteps", 5)

    # Load fftw-wisdom if possible
    if use_wisdom and os.path.exists(wisdom_file):
        pyfftw.import_wisdom(load_wisdom(wisdom_file))

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
    time_start = time.perf_counter()
    if not channels:
        field = ExternalField(fields_params, grid_xyz, nthreads=pyfftw_threads)
    else:
        field = ProbePumpField(
            fields_params,
            grid_xyz,
            probe_pump_idx=probe_pump_idx,
            nthreads=pyfftw_threads,
        )
    time_field_setup = time.perf_counter()
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
    time_vacem_setup = time.perf_counter()
    time_integral = vacem.calculate_amplitudes(grid_t, save_path=amplitudes_file)
    time_amplitudes = time.perf_counter()
    maxrss_amplitudes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info("MILESTONE: Amplitudes are calculated")

    del field, vacem

    # Calculate spectra
    if do_postprocess:
        postprocess_print = get_postprocess_info(postprocess_params)
        logger.info(postprocess_print)
        analyzer = VacuumEmissionAnalyzer(
            fields_params, data_path=amplitudes_file, save_path=spectra_file
        )
        analyzer.get_spectra(
            perp_field_idx=perp_field_idx,
            perp_type=perp_type,
            calculate_spherical=calculate_spherical,
            spherical_params=spherical_params,
            calculate_discernible=calculate_discernible,
        )
        logger.info("MILESTONE: Spectra are calculated from amplitudes")
    time_postprocess = time.perf_counter()

    # Save gained wisdom (for fftw)
    save_wisdom(ini_file, wisdom_file)

    time_per_iteration = (time_amplitudes - time_field_setup) / len(grid_t)
    # Performance estimation
    timings = {
        "start": time_start,
        "field_setup": time_field_setup,
        "vacem_setup": time_vacem_setup,
        "amplitudes": time_amplitudes,
        "postprocess": time_postprocess,
        "per_iteration": time_per_iteration,
        "total": time_postprocess - time_start,
    }

    maxrss_total = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    memory = {"maxrss_amplitudes": maxrss_amplitudes, "maxrss_total": maxrss_total}

    perf_stats = {"timings": timings, "memory": memory}

    write_yaml(performance_file, perf_stats)

    perf_print = get_performance_stats(perf_stats)
    print(perf_print)
    logger.info(perf_print)

    if test_run:
        time_overhead = time_amplitudes - time_field_setup - time_integral
        time_per_iteration = time_integral / len(grid_t)
        expected_time = time_per_iteration * expected_timesteps
        time_total = time_overhead + expected_time
        test_run_str_print = test_run_str.format(
            format_time(time_per_iteration),
            format_time(time_overhead),
            format_time(time_total),
        )
        logger.info(test_run_str_print)
        print(test_run_str_print)

    print("Simulation finished!")

    # End time
    time_log_end = time.asctime(time.localtime())
    end_print = simulation_end_str.format(time_log_end)
    logger.info(end_print)


if __name__ == "__main__":
    args = parse_args()
    quvac_simulation(args.input, args.output, args.wisdom)
