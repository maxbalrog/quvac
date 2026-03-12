"""
Useful generic utilities.
"""

import gc
import importlib
import inspect
import math
import os
from pathlib import Path
import pkgutil
import platform
import resource
import shutil
import sys

import numpy as np
import pyfftw
import yaml

from quvac.grid import setup_grids


def read_yaml(yaml_file):
    """
    Read a YAML file and return its contents.

    Parameters
    ----------
    yaml_file : str
        Path to the YAML file.

    Returns
    -------
    dict
        Contents of the YAML file.
    """
    with open(yaml_file) as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)
            return exc


def write_yaml(yaml_file, data):
    """
    Write data to a YAML file.

    Parameters
    ----------
    yaml_file : str
        Path to the YAML file.
    data : dict
        Data to write to the YAML file.
    """
    with open(yaml_file, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def format_time(seconds):
    """
    Format time in seconds to a human-readable string.

    Parameters
    ----------
    seconds : float
        Time in seconds.

    Returns
    -------
    str
        Formatted time string.
    """
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    out_str = [
        f"{days:.0f} days" * bool(days),
        f"{hours:.0f} h" * bool(hours),
        f"{minutes:.0f} min" * bool(minutes),
        f"{seconds:.2f} s",
    ]
    return " ".join(out_str)


def format_memory(mem):
    """
    Format memory in kilobytes to a human-readable string.

    Parameters
    ----------
    mem : float
        Memory in kilobytes.

    Returns
    -------
    str
        Formatted memory string.
    """
    units = "KB MB GB TB".split()
    idx = 0
    while mem > 1024:
        mem /= 1024
        idx += 1
    return f"{mem:.2f} {units[idx]}"


def save_wisdom(ini_file, wisdom_file=None, add_host_name=False):
    """
    Save FFTW wisdom to a file.

    Parameters
    ----------
    ini_file : str
        Path to the initialization file.
    wisdom_file : str, optional
        Path to save the FFTW wisdom, by default None.
    add_host_name : bool, optional
        Whether to add the host name to the wisdom file name, by default False.
    """
    if wisdom_file is None:
        wisdom_path = os.path.dirname(ini_file)
        if not os.path.exists(wisdom_path):
            Path(wisdom_path).mkdir(parents=True, exist_ok=True)
        wisdom_name = "fftw-wisdom"
        if add_host_name:
            wisdom_name += platform.node()
        wisdom_file = os.path.join(wisdom_path, wisdom_name)
    else:
        wisdom_path = os.path.dirname(wisdom_file)
        if not os.path.exists(wisdom_path):
            Path(wisdom_path).mkdir(parents=True, exist_ok=True)
    wisdom = pyfftw.export_wisdom()
    with open(wisdom_file, "wb") as f:
        f.write(b"\n".join(wisdom))


def load_wisdom(wisdom_file):
    """
    Load FFTW wisdom from a file.

    Parameters
    ----------
    wisdom_file : str
        Path to the FFTW wisdom file.

    Returns
    -------
    tuple
        FFTW wisdom.
    """
    with open(wisdom_file, "rb") as f:
        wisdom = f.read()
    return tuple(wisdom.split(b"\n"))


def get_maxrss():
    """
    Get the maximum resident set size used (in kilobytes).

    Returns
    -------
    int
        Maximum resident set size used.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def zip_directory_shutil(directory_path, output_path):
    """
    Zip a directory using shutil.

    Parameters
    ----------
    directory_path : str
        Path to the directory to zip.
    output_path : str
        Path to save the zipped directory.
    """
    shutil.make_archive(output_path, 'zip', directory_path)


def find_classes_in_package(package_name):
    """
    Find all class names in a given package.

    Parameters
    ----------
    package_name : str
        Name of the package.

    Returns
    -------
    list of str
        List of class names in the package.
    """
    classes = []
    
    # Import the package
    package = importlib.import_module(package_name)
    
    # Recursively find all modules in the package
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, 
                                                   package.__name__ + "."):
        try:
            module = importlib.import_module(module_name)
            
            # Inspect module members and find classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Ensure class belongs to the module (not an imported one)
                if obj.__module__ == module_name:
                    classes.append(f"{module_name}.{name}")
        
        except Exception as e:
            print(f"Skipping {module_name} due to error: {e}")

    return classes


def round_to_n(x, n):
    """
    Round up to n significant digits.

    Parameters
    ----------
    x : int or float
        Number to round up.
    n : int
        Number of digits to round up to.

    Returns
    -------
    int or float:
        Rounded number.
    """
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)


def size_to_Gb(size):
    """
    Convert the size of float64 array to GBs.
    """
    return size*8 / 1024**3


def estimate_max_required_memory(size):
    """
    Estimate max requred memory based on the grid size.
    """
    # this value is estimated by running the simulation with different grid sizes,
    # looking at the max used memory and fitting a line to the dependency
    # max memory vs grid size
    MEMORY_SCALING = 56
    estimated_mem = size_to_Gb(math.prod(size))*MEMORY_SCALING

    # memory buffer just in case
    SAFE_BUFFER = 10
    return int(np.ceil(estimated_mem + SAFE_BUFFER))


def estimate_memory_usage(ini_file):
    """
    Estimate potential memory usage for a given ini file.

    Parameters
    ----------
    ini_file: str
        Path to the initialization file

    Returns
    -------
    str
        Required memory in format '<number>GB'.
    """
    ini_config = read_yaml(ini_file)
    grid_xyz, _ = setup_grids(
        ini_config.get("fields", None), 
        ini_config.get("grid", None),
    )

    required_memory = estimate_max_required_memory(grid_xyz.grid_shape)
    return f"{required_memory}GB"


def free_memory():
    gc.collect()
    if sys.platform == "linux":
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
