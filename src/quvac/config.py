"""
Configuration file for simulations. 

``FDTYPE`` and ``CDTYPE`` control the float precision for the 
calculations (``float32`` or ``float64``).

``DEFAULT_SUBMITIT_PARAMS`` is a dictionary with default parameters for
job submission to the cluster.
"""

DEFAULT_SUBMITIT_PARAMS = {
    "slurm_partition": "hij-gpu",
    "cpus_per_task": 16,
    "slurm_mem": "20GB",
    "timeout_min": 240,
}

FDTYPE = "float64"
CDTYPE = "complex128"

RUST_TOML = "/home/maximus/Research/github/quantum-vacuum-rs/Cargo.toml"
