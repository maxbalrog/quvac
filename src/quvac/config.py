"""
Configuration file for simulations. 

``FDTYPE`` and ``CDTYPE`` control the float precision for the 
calculations (``float32`` or ``float64``).

``DEFAULT_SLURM_PARAMS`` is a dictionary with default parameters for
job submission to the cluster.
"""

DEFAULT_SLURM_PARAMS = {
    "slurm_partition": "hij-gpu",
    "cpus_per_task": 16,
    "memory": "20GB",
    "walltime": "1:00:00",
}

FDTYPE = "float64"
CDTYPE = "complex128"

FFTW_FLAG = "FFTW_MEASURE"
