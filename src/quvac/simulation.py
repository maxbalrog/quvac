#!/usr/bin/env python3
'''
Here we provide a script to launch Vacuum Emission simulation,
do postprocessing and measure performance
'''

import argparse
import logging
import os
from pathlib import Path
import time
import resource

import numpy as np
import numexpr as ne
import pyfftw

from quvac.field.external_field import ExternalField, ProbePumpField
from quvac.integrator.vacuum_emission import VacuumEmission
from quvac.grid import setup_grids
from quvac.postprocess import VacuumEmissionAnalyzer
from quvac.utils import (read_yaml, write_yaml, format_memory, format_time,
                         load_wisdom, save_wisdom)

logger = logging.getLogger(__name__)

# ini yaml structure
'''
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
'''

# timings structure
performance_str = '''
Timings:
=================================================
Field setup:               {:>15s}
Vacem setup:               {:>15s}
Amplitudes calculation:    {:>15s}
Postprocess:               {:>15s}
-------------------------------------------------
Per iteration:             {:>15s}  
-------------------------------------------------
Total:                     {:>15s}
=================================================

Memory (max usage):
=================================================
Amplitudes calculation:    {:>15s}
Total:                     {:>15s}
=================================================
'''


def get_performance_stats(perf_stats):
    timings = perf_stats['timings']
    timings = {
        'field_setup': timings['field_setup']-timings['start'],
        'vacem_setup': timings['vacem_setup']-timings['field_setup'],
        'amplitudes': timings['amplitudes']-timings['vacem_setup'],
        'postprocess': timings['postprocess']-timings['amplitudes'],
        'per_iteration': timings['per_iteration'],
        'total': timings['postprocess']-timings['start'],
    }
    timings = {k: format_time(t) for k,t in timings.items()}
    memory = {k: format_memory(m) for k,m in perf_stats['memory'].items()}
    perf_print = performance_str.format(timings['field_setup'],
                                        timings['vacem_setup'],
                                        timings['amplitudes'],
                                        timings['postprocess'],
                                        timings['per_iteration'],
                                        timings['total'],
                                        memory['maxrss_amplitudes'],
                                        memory['maxrss_total'])
    return perf_print 


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


def quvac_simulation(ini_file, save_path=None, wisdom_file='wisdom/fftw-wisdom'):
    '''
    Launch a single quvac simulation for given <ini>.yaml file

    Parameters:
    -----------
    ini_file: str (format <path>/<file_name>.yaml)
        Initial configuration file containing all simulation parameters
    save_path: str
        Path to save simulation results to
    '''
    # Check that ini file and save_path exists
    assert os.path.isfile(ini_file), f"{ini_file} is not a file or does not exist"
    if save_path is None:
        save_path = os.path.dirname(ini_file)
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
    amplitudes_file = os.path.join(save_path, 'amplitudes.npz')
    spectra_file = os.path.join(save_path, 'spectra.npz')
    
    # Setup logger
    logger_file = os.path.join(save_path, 'simulation.log')
    logging.basicConfig(filename=logger_file, encoding='utf-8', level=logging.DEBUG,
                        format=f'%(asctime)s %(message)s')

    # Load and parse ini yaml file
    ini_config = read_yaml(ini_file)
    fields_params = ini_config["fields"]
    if isinstance(fields_params, dict):
        fields_params = list(fields_params.values())
    grid_params = ini_config["grid"]
    perf_params = ini_config.get("performance", {})

    # Determine integrator type
    integrator_params = ini_config.get('integrator', {})
    integrator_type = integrator_params.get('type', 'vacuum_emission')
    channels = integrator_type.endswith('channels')
    if channels:
        probe_pump_idx = integrator_params.get('probe_pump_idx', None)
    
    # Determine postprocess steps
    postprocess_params = ini_config.get('postprocess', {})
    calculate_spherical = postprocess_params.get('calculate_spherical', False)
    spherical_params = postprocess_params.get('spherical_params', {})
    calculate_discernible = postprocess_params.get('calculate_discernible', False)
    perp_type = postprocess_params.get('perp_polarization_type', None)
    perp_field_idx = postprocess_params.get('perp_field_idx', 1)
    
    # Set up number of threads
    nthreads = perf_params.get('nthreads', os.cpu_count())
    ne.set_num_threads(nthreads)
    pyfftw.config.NUM_THREADS = nthreads

    # Load fftw-wisdom if possible
    if os.path.exists(wisdom_file):
        pyfftw.import_wisdom(load_wisdom(wisdom_file))

    # Get grids
    grid_xyz, grid_t = setup_grids(fields_params, grid_params)
    logger.info("Grids are created")

    # Field setup
    time_start = time.perf_counter()
    if not channels:
        field = ExternalField(fields_params, grid_xyz, nthreads=nthreads)
    else:
        field = ProbePumpField(fields_params, grid_xyz, probe_pump_idx=probe_pump_idx,
                               nthreads=nthreads)
    time_field_setup = time.perf_counter()
    logger.info("Fields are set up")

    # Calculate amplitudes
    vacem = VacuumEmission(field, grid_xyz, nthreads, channels=channels)
    time_vacem_setup = time.perf_counter()
    vacem.calculate_amplitudes(grid_t, save_path=amplitudes_file)
    time_amplitudes = time.perf_counter()
    maxrss_amplitudes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info("Amplitudes are calculated")

    del field, vacem

    # Calculate spectra
    analyzer = VacuumEmissionAnalyzer(fields_params, data_path=amplitudes_file,
                                      save_path=spectra_file)
    analyzer.get_spectra(perp_field_idx=perp_field_idx,
                         perp_type=perp_type,
                         calculate_spherical=calculate_spherical,
                         spherical_params=spherical_params,
                         calculate_discernible=calculate_discernible)
    time_postprocess = time.perf_counter()
    logger.info("Spectra calculated from amplitudes")

    # Save gained wisdom (for fftw)
    save_wisdom(ini_file, wisdom_file)

    time_per_iteration = (time_amplitudes - time_field_setup)/len(grid_t)
    # Performance estimation
    timings = {
        'start': time_start,
        'field_setup': time_field_setup,
        'vacem_setup': time_vacem_setup,
        'amplitudes': time_amplitudes,
        'postprocess': time_postprocess,
        'per_iteration': time_per_iteration
    }

    maxrss_total = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    memory = {
        'maxrss_amplitudes': maxrss_amplitudes,
        'maxrss_total': maxrss_total
    }

    perf_stats = {
        'timings': timings,
        'memory': memory
    }

    perf_print = get_performance_stats(perf_stats)
    print(perf_print)
    logger.info(perf_print)

    print("Simulation finished!")


if __name__ == '__main__':
    args = parse_args()
    quvac_simulation(args.input, args.output, args.wisdom)