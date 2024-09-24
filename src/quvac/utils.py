'''
Useful generic utilities
'''

import os
from pathlib import Path
import platform
import yaml

import pyfftw


def read_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as exc:
            print(exc)
            return exc


def write_yaml(yaml_file, data):
    with open(yaml_file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def format_time(seconds):
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    out_str = [f'{days:.0f} days'*bool(days),
               f'{hours:.0f} h'*bool(hours),
               f'{minutes:.0f} min'*bool(minutes),
               f'{seconds:.2f} s']
    return ' '.join(out_str)


def format_memory(mem):
    '''
    mem: float
        Memory in KB (kilobyte)
    '''
    units = 'KB MB GB TB'.split()
    idx = 0
    while mem > 1024:
        mem /= 1024
        idx += 1
    return f'{mem:.2f} {units[idx]}'


def save_wisdom(ini_file, wisdom_file=None, add_host_name=False):
    if wisdom_file is None:
        wisdom_path = os.path.dirname(ini_file)
        if not os.path.exists(wisdom_path):
            Path(wisdom_path).mkdir(parents=True, exist_ok=True)
        wisdom_name = 'fftw-wisdom' 
        if add_host_name:
            wisdom_name += platform.node() 
        wisdom_file = os.path.join(wisdom_path, wisdom_name)
    else:
        wisdom_path = os.path.dirname(wisdom_file)
        if not os.path.exists(wisdom_path):
            Path(wisdom_path).mkdir(parents=True, exist_ok=True)
    wisdom = pyfftw.export_wisdom()
    with open(wisdom_file, 'wb') as f:
        f.write(b'\n'.join(wisdom))


def load_wisdom(wisdom_file):
    with open(wisdom_file, 'rb') as f:
        wisdom = f.read()
    return tuple(wisdom.split(b'\n'))
