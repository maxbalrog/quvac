"""
Useful generic utilities.
"""

import pkgutil
import importlib
import inspect
import os
import platform
import resource
from pathlib import Path
import shutil

import numpy as np
import pyfftw
import yaml


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
    with open(yaml_file, "r") as stream:
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
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
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
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)
