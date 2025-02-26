#!/usr/bin/env python3
"""
Script to create images of participating fields
"""
import argparse
from copy import deepcopy
import gc
import os
import time
from pathlib import Path

from quvac.field.external_field import ExternalField
from quvac.grid import setup_grids
from quvac.plotting import plot_fields
from quvac.simulation import set_precision
from quvac.utils import read_yaml


def parse_args():
    description = "Plot fields present in the simulation"
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument(
        "--input", "-i", default=None, help="Input yaml file with field and grid params"
    )
    argparser.add_argument(
        "--output", "-o", default=None, help="Path to save simulation data to"
    )
    return argparser.parse_args()


def image_fields(ini_file, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(ini_file), 'imgs')
    Path(save_path).mkdir(parents=True, exist_ok=True)

    ini_config = read_yaml(ini_file)
    fields_params = ini_config['fields']
    grid_params = ini_config["grid"]

    perf_params = ini_config.get("performance", {})
    precision = perf_params.get("precision", "float32")
    set_precision(precision)

    # Setup grids
    grid_xyz, grid_t = setup_grids(deepcopy(fields_params), deepcopy(grid_params))
    grid_xyz.get_k_grid()
    grid_shape = grid_xyz.grid_shape
    print(f"Grid shape: {grid_shape}")
    print(f"Time steps: {len(grid_t)}")

    plot_keys = ["Intensity"]
    # Setup fields and plot
    for idx,field_param in enumerate(fields_params.values()):
        field = ExternalField([field_param], grid_xyz)
        for t0 in [0, grid_t[-1]]:
            save_loc = os.path.join(save_path, f"field_{idx+1}_t0_{t0*1e15:.1f}fs")
            plot_fields(field, t=t0, plot_keys=plot_keys, norm_lim=1e-10, save_path=save_loc)
        del field
        gc.collect()
    print("Plotting finished successfully")
    

if __name__ == "__main__":
    args = parse_args()
    image_fields(args.input, args.output)