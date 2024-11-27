from quvac.utils import format_memory, format_time


# Warning when total signal on spherical grid differs from
# cartesian grid one
sph_interp_warn = """WARNING:
{:s} signal on cartesian and spherical grid differ by more than 1%:
    N total (xyz): {:.3f}
    N total (sph): {:.3f}
"""

# timings structure
performance_str = '''
Timings:
====================================================
Field setup:               {:>15s}
Vacem setup:               {:>15s}
Amplitudes calculation:    {:>15s}
    Per time step:         {:>15s}
Postprocess:               {:>15s}
----------------------------------------------------
Total:                     {:>15s}
====================================================

Memory (max usage):
====================================================
Amplitudes calculation:    {:>15s}
Total:                     {:>15s}
====================================================
'''

# timings for parallel
performance_parallel_str = '''
Timings:
====================================================
Run jobs:                  {:>15s}
Postprocess:               {:>15s}
----------------------------------------------------
Total:                     {:>15s}
====================================================

Memory (max usage):
====================================================
Running jobs:              {:>15s}
Total:                     {:>15s}
====================================================
'''

# grid params
grid_str = '''
Grid:
====================================================
Space

Number of points (x,y,z):  {:>25}
Box for x axis:            {:>25}
Box for y axis:            {:>25}
Box for z axis:            {:>25}
----------------------------------------------------
Time

Number of points:          {:>25}
Box:                       {:>25}
====================================================
'''


postprocess_str = '''
Postprocess parameters
====================================================
Perp polarization type:           {:>15}
Perp field idx:                   {:>15}
Signal on spherical grid:         {:>15}
Discernible signal:               {:>15}
====================================================
'''

test_run_str = '''
====================================================
Time per iteration:               {:>15s}
Overhead time:                    {:>15s}  
Estimated time (full run):        {:>15s}
====================================================
'''


simulation_start_str = '''
####################################################
Start of simulation:    {:>20}
####################################################
'''


simulation_end_str = '''
####################################################
End of simulation:      {:>20}
####################################################
'''


def get_grid_params(grid_xyz, grid_t):
    nx, ny, nz = grid_xyz.grid_shape
    grid_xyz_size = f'({nx}, {ny}, {nz})'
    x, y, z = grid_xyz.grid
    x_start, x_end = x[0]*1e6, x[-1]*1e6
    x_box = f'({x_start:.2f}, {x_end:.2f}) micron'
    y_start, y_end = y[0]*1e6, y[-1]*1e6
    y_box = f'({y_start:.2f}, {y_end:.2f}) micron'
    z_start, z_end = z[0]*1e6, z[-1]*1e6
    z_box = f'({z_start:.2f}, {z_end:.2f}) micron'

    grid_t_size = len(grid_t)
    t_start, t_end = grid_t[0]*1e15, grid_t[-1]*1e15
    t_box = f'({t_start:.2f}, {t_end:.2f}) fs'

    grid_print = grid_str.format(grid_xyz_size,
                                 x_box,
                                 y_box,
                                 z_box,
                                 grid_t_size,
                                 t_box)
    return grid_print


def get_postprocess_info(postprocess_params):
    calculate_spherical = postprocess_params.get('calculate_spherical', False)
    spherical_params = postprocess_params.get('spherical_params', {})
    calculate_discernible = postprocess_params.get('calculate_discernible', False)
    perp_type = postprocess_params.get('perp_polarization_type', None)
    perp_field_idx = postprocess_params.get('perp_field_idx', 1)

    postprocess_print = postprocess_str.format(perp_type,
                                               perp_field_idx,
                                               str(calculate_spherical),
                                               str(calculate_discernible))
    return postprocess_print


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
                                        timings['per_iteration'],
                                        timings['postprocess'],
                                        timings['total'],
                                        memory['maxrss_amplitudes'],
                                        memory['maxrss_total'])
    return perf_print 


def get_parallel_performance_stats(perf_stats):
    timings = perf_stats['timings']
    timings = {
        'jobs': timings['jobs']-timings['start'],
        'postprocess': timings['postprocess']-timings['jobs'],
        'total': timings['postprocess']-timings['start'],
    }
    timings = {k: format_time(t) for k,t in timings.items()}
    memory = {k: format_memory(m) for k,m in perf_stats['memory'].items()}
    perf_print = performance_parallel_str.format(timings['jobs'],
                                                timings['postprocess'],
                                                timings['total'],
                                                memory['maxrss_jobs'],
                                                memory['maxrss_total'])
    return perf_print 