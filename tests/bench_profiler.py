"""
Test simulation is run and profiler reports are automatically generated.

Currently two profilers are used:
1. `scalene` to study the time performance with line-by-line timings.
2. `memray` to study memory consumption.
"""

import os
from pathlib import Path

import pytest

from quvac.utils import read_yaml, write_yaml
from tests.config_for_tests import PROFILER_CONFIG_PATH, SIMULATION_SCRIPT


@pytest.mark.benchmark
def run_scalene_and_memray(path, ini_data):
    current_dir = os.path.abspath(os.getcwd())

    Path(path).mkdir(parents=True, exist_ok=True)

    ini_file = os.path.join(path, "ini.yml")
    write_yaml(ini_file, ini_data)

    profile_folder = os.path.join(path, "profiler_info")
    Path(profile_folder).mkdir(parents=True, exist_ok=True)
    scalene_file = "report-scalene.json"
    scalene_path = os.path.join(profile_folder, scalene_file)
    memray_file = os.path.join(profile_folder, "report-memray.bin")

    if os.path.isfile(scalene_file):
        os.remove(scalene_file)
    if os.path.isfile(memray_file):
        os.remove(memray_file)

    # Run scalene profiler
    status = os.system(f"scalene run -o {scalene_path} --cpu-only "
                       f"{SIMULATION_SCRIPT} --input {ini_file}")
    assert status == 0, "Scalene execution did not finish successfully"
    os.system(f"cd {profile_folder} && scalene view --standalone {scalene_file}")
    os.system(f"cd {current_dir}")

    # Run memray profiler
    status = os.system(f"memray run -o {memray_file} "
                       f"{SIMULATION_SCRIPT} --input {ini_file}")
    assert status == 0, "Memray execution did not finish successfully"
    os.system(f"memray flamegraph {memray_file}")


@pytest.mark.benchmark
def test_scalene_and_memray():
    # Load default simulation parameters
    ini_data = read_yaml(PROFILER_CONFIG_PATH)

    path = "data/profiler"
    run_scalene_and_memray(path, ini_data)