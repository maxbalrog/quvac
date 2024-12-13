"""
Here useful constants are stored that configure simulations
"""

DEFAULT_SUBMITIT_PARAMS = {
    "slurm_partition": "hij-gpu",
    "cpus_per_task": 16,
    "slurm_mem": "20GB",
    "timeout_min": 240,
}

FDTYPE = "float64"
CDTYPE = "complex128"
