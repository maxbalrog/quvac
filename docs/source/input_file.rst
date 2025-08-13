Create input file
=================

Overview
--------------
We use ``.yml`` format for input files which has a dictionary-like structure. It has the following main sections:
    - ``mode``: str (optional), one of ``simulation``, ``postprocess`` or (default) ``simulation_postprocess``
            Type of calculation to perform.

    - ``fields``: dict of dicts
            Parameters of background fields.

    - ``grid``: dict
            Grid parameters.

    - ``integrator``: dict
            Type of amplitude to calculate (total vacuum emission or linearized in the probe field).

    - ``performance``: dict
            Performance-related parameters.

    - ``postprocess``: dict
            Postprocessing parameters (which observables to calculate from the complex amplitudes).

``quvac-simulation-parallel`` additionally requires section ``cluster_params`` describing
parameters of parallelization. Similarly, ``quvac-gridscan`` requires a section ``variables`` (parameters
being scanned and slurm job parameters), ``quvac-optimization`` requires a section ``optimization``
(parameters to setup optimization experiment: optimized variables, metrics, sampling strategy, ...). 


Fields
--------------
This section of ``.yml`` file is constructed as ``{'field_1': {...}, 'field_2': {...}, ...}`` where each dictionary of field parameters has the ``field_type`` parameter and 
other parameters specific to the chosen field type. 

``field_type`` is constructed via the combination of the field name (``dipole``, ``paraxial_gaussian``, ``laguerre_gaussian``) 
and how it is simulated (``analytic``, ``maxwell``). For instance, to use the analytic expression of paraxial Gaussian for all time steps, choose ``paraxial_gaussian_analytic``;
to use the dipole pulse expression for initialization and use linear Maxwell equations for later time steps, choose ``dipole_maxwell``.

Currently there are two special fields that are supported only as an analytic expression: ``eb_inhomogeneity``, ``plane_wave``.

For the full list of available keywords, refer to ``quvac.field.ANALYTIC_FIELDS`` and ``quvac.field.SPATIAL_MODEL_FIELDS``.

+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|                   |                                                                         Field types                                                                          |
| Field parameters  +----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|                   |                      dipole                        |                   paraxial_gaussian                |                  laguerre_gaussian                 |
+===================+====================================================+====================================================+====================================================+
|   ``focus_x``     |                                                                     Focus location in space                                                                  |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|   ``focus_t``     |                                                                      Focus location in time                                                                  |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|      ``W``        |                                                                            Energy                                                                            |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|     ``lam``       |                                                                          Wavelength                                                                          |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|     ``tau``       |                                                                           Duration                                                                           |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|``theta``, ``phi`` |        Virtual dipole moment orientation           |                                          Optical axis orientation                                       |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|    ``beta``       |                       --                           |                                             Polarization angle                                          |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|   ``phase0``      |                       --                           |                                          Phase delay at the focus                                       |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|     ``w0``        |                       --                           |                                               Waist size                                                |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|    ``order``      |                       --                           |                                          Paraxial expansion order                                       |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|     ``E0``        |                       --                           |                                   Field amplitude (if W is not specified)                               |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|     ``p``         |                       --                           |                         --                         |      Radial index of the Laguerre-Gaussian mode    |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|     ``l``         |                       --                           |                         --                         |    Azimuthal index of the Laguerre-Gaussian mode   |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|  ``dipole_type``  |           ``electric`` or ``magnetic``             |                                                   --                                                    |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+
|   ``envelope``    |  Temporal envelope type (``plane`` or ``gauss``)   |                                                   --                                                    |
+-------------------+----------------------------------------------------+----------------------------------------------------+----------------------------------------------------+

Grid
--------------
Required keys are:
    -  ``mode`` : str
        Mode of grid creation (``dynamic`` or ``static``).

Keys for ``static`` mode:
    - ``box_xyz`` : tuple of float
        Box size for the spatial grid.

    - ``Nxyz`` : tuple of int
        Number of grid points along each spatial dimension.
    
    - ``Nt`` : int
        Number of temporal points.
    
    - ``box_t`` : float or tuple of float
        Time duration or start and end times for the temporal grid.

Keys for ``dynamic`` mode:
    - ``collision_geometry`` : str
        Specifies the collision geometry ('x', 'y', 'z').

    - ``transverse_factor`` : float
        Factor to scale the transverse size.
    
    - ``longitudinal_factor`` : float
        Factor to scale the longitudinal size.
    
    - ``time_factor`` : float
        Factor to scale the time duration.
    
    - ``spatial_resolution`` : float or list of float, optional
        Controls the spatial resolution.
    
    - ``time_resolution`` : float, optional
        Controls the temporal resolution.
    
    - ``ignore_idx`` : list of int, optional
        Indices of fields to ignore for dynamic grid creation.


Integrator (optional)
---------------------
Keys:
    - ``type``: str
        ``vacuum_emission`` (calculate the total vacuum emission amplitude) or ``vacuum_emission_channels`` (calculate the amplitude linearized in the probe field)
    - ``probe_pump_idx``: dict
        Indices of probe and pump fields.
            - ``probe``: list of int
                Indices of the probe field, by default [0].
            - ``pump``: list of int
                Indices of the pump field, by default [1].

Performance (optional)
----------------------
Keys:
    - ``precision``: str
        Numerical precision for calculations: ``float32`` or (by default) ``float64``.
    - ``nthreads``: int
        Number of threads to use for ``numexpr`` library, by default all available CPUs.
    - ``pyfftw_threads``: int
        Number of threads to use for ``pyfftw`` library, by default equal to ``nthreads``.
    - ``test_run``: bool
        Whether to do a test run to estimate the resources for the full calculation.
    - ``test_timesteps``: int
        Number of timesteps for a test run, by default 5.
    - ``use_wisdom``: bool
        Whether to use existing wisdom file for ``pyfftw`` planning.


Postprocessing (optional)
-------------------------
This section is relevant only when ``mode`` is ``postprocess`` or ``simulation_postprocess``. Relevant keys for the polarization-insensitive signals:
    - ``calculate_xyz_background`` : bool, optional
        Whether to calculate the background spectra on Cartesian grid, 
        by default False.
    - ``bgr_idx`` : int, optional
        Index of the background field, by default None.
    - ``calculate_spherical`` : bool, optional
        Whether to calculate the spectra on spherical grid,
        by default False.
    - ``spherical_params`` : dict, optional
        Parameters for the spherical grid, by default None.
    - ``calculate_discernible`` : bool, optional
        Whether to calculate the discernible signal, by default False.
    - ``discernibility`` : str, optional
        Type of discernibility, (by default) ``angular`` or ``spectral``.

Relevant keys for the polarization-sensitive signals:
    - ``perp_field_idx`` : int, optional
        Index of the perpendicular field, by default 1.
    - ``perp_type`` : str, optional
        Type of perpendicular polarization, ``optical_axis`` or ``local_axis``, by default None.
    - ``calculate_spherical`` : bool, optional
        Whether to calculate the spectra on spherical grid,
        by default False.
    - ``spherical_params`` : dict, optional
        Parameters for the spherical grid, by default None.
    - ``stokes`` : bool, optional
        Whether to calculate Stokes parameters, by default False.

Cluster_params (for ``quvac-simulation-parallel``)
--------------------------------------------------
Keys:
    - ``n_jobs``: int
        Number of jobs to parallelize between, by default 2.
    - ``max_jobs``: int
        Maximal number of jobs to submit simultaneously, by default equal to ``n_jobs``.
    - ``cluster``: str,
        Where perform calculations, ``local`` or ``slurm``.
    - ``sbatch_params``: dict
        Submission parameters for a single job. Possible keys: ``slurm_partition``,
        ``cpus_per_task``, ``slurm_mem``, ``timeout_min``. By default, 
        ``quvac.config.DEFAULT_SUBMITIT_PARAMS``.

Variables (for ``quvac-gridscan``)
----------------------------------
Keys:
    - ``create_grids``: bool
        Flag to create grids given [start, end, n_steps].
    - ``fields``: dict
        Parameters over which to perform grid scan.
    - ``cluster``: dict
        - ``cluster``: str
            ``local`` or ``slurm``.
        - ``max_parallel_jobs``: int
            Maximal number of submitted jobs in parallel.
        - ``sbatch_params``: dict
            Submission parameters for a single job.

Optimization (for ``quvac-optimization``)
-----------------------------------------
Keys:
    - ``name``: str
        Optimization name.
    - ``parameters``: dict
        Optimized parameters given as lists [start, end].
    - ``energy_fields``: dict
        Fields participating in the energy optimization (to check that the fixed budget is not violated):
            - ``fields``: list of int
                Fields for which the fixed energy budget constraint should be satisfied.
            - ``optimized_fields``: list of int
                Fields being optimized.
    - ``scales``: dict
        Scales for optimized parameters. For instance, parameter could be the duration with bounds 
        [20,50] and the provided scale 1e-15 corresponding to femtoseconds.
    - ``cluster``: dict
        Submission parameters for a single job.
    - ``n_trials``: int
        Number of trials to perform.
    - ``objectives``: list of [str, bool] 
        Objective functions, for each function specify its name and whether to minimize it.
        For instance, for the total discernible signal objective funtion corresponds to ``[['N_disc', False]]``.
    - ``objectives_params``: dict
        Objective parameters:
            - ``detectors``: dict or list of dicts
                Detector parameters (``phi0``, ``theta0`` for the center of detector, ``dphi``, ``dtheta`` for its size).
    - ``track_metrics``: list of str
        Additional metrics to track.
    - ``parameter_constraints``: list of str
        Optimized parameter constraints, for example ``a + b + c <= 1``.
    - ``gs_params``: dict
        Generation strategy parameters:
            - ``num_random_trials``: int
                Number of random trials to initialize Gaussian process.
        

