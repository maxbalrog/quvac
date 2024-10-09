'''
This script provides utilities for grid definition
'''

import numpy as np
from scipy.constants import pi, c
import pyfftw

from quvac.grid import GridXYZ


def get_ek(theta, phi):
    ek = np.array([np.sin(theta) * np.cos(phi),
                   np.sin(theta) * np.sin(phi),
                   np.cos(theta)])
    return ek


def get_pol_basis(theta, phi):
    e1 = np.array([np.cos(theta) * np.cos(phi),
                   np.cos(theta) * np.sin(phi),
                   -np.sin(theta)])
    e2 = np.array([-np.sin(phi),
                   np.cos(phi),
                   0])
    return e1, e2


def get_kmax_grid(field_params):
    '''
    Calculates maximum k-vector along every dimension

    Laser params: dict
        Required keys: lam, tau, w0/w0x, theta, phi
        theta, phi in degrees
    '''
    required_keys = 'lam tau theta phi'.split()
    err_msg = f"Field parameters must have {required_keys} keys"
    assert all(key in field_params for key in required_keys), err_msg

    lam, tau = field_params['lam'], field_params['tau']
    theta, phi = field_params['theta'], field_params['phi']
    if 'w0' in field_params:
        w0 = field_params['w0']
    elif 'w0x' in field_params:
        w0 = min(field_params['w0x'], field_params['w0y'])
    
    k = 2*np.pi/lam    
    theta *= pi/180
    phi = phi*pi/180 if np.sin(theta) != 0. else 0.
    
    ek = get_ek(theta, phi)
    e1, e2 = get_pol_basis(theta, phi)
        
    kbw_perp = 4/w0
    kbw_long = 8/(c*tau)
    
    k0 = ek * k
    kmax = np.abs(k0 + ek * kbw_long)
    
    for beta in np.linspace(0, 2*np.pi, 64, endpoint=False):
        kp = k0 + ek * kbw_long + kbw_perp * (np.cos(beta) * e1 + np.sin(beta) * e2)
        kmax = np.maximum(kmax, np.abs(kp))
    
    return kmax


def get_xyz_size(fields, box_size, grid_res=1, equal_resolution=False):
    '''
    Calculates necessary spatial resolution
    
    fields: list of dicts
        List of fields' parameters
    box_size: typing.Sequence (e.g., list, np.array)
        Box size for 3 dimensions
    grid_res: float
        Controls the resolution
    equal_resolution: bool
        Flag for equal resolution for every dimension
    '''
    box_size = np.array(box_size)

    kmax = np.zeros((3,))
    for field_params in fields:
        kmax = np.maximum(kmax, get_kmax_grid(field_params))
    
    # if a square box is required
    if equal_resolution:
        kmax = np.max(kmax) * np.ones(3)
    
    N = np.ceil(grid_res * box_size * 3 * kmax/pi).astype(int)
    N = [pyfftw.next_fast_len(n) for n in N]
    return N


def get_t_size(t_start, t_end, lam, grid_res=1):
    '''
    Calculates necessary temporal resolution
    
    t_start, t_end: floats
        Start time, end time
    lam: float
        Wavelength
    grid_res: float
        Controls the resolution
    '''
    fmax = c/lam
    return int(np.ceil((t_end-t_start)*fmax*6*grid_res))


def create_dynamic_grid(fields_params, grid_params):
    pass


def setup_grids(fields_params, grid_params):
    '''
    Create spatial and temporal grids from given sizes or
    dynamically from field parameters (e.g., tau, w0)
    '''
    if grid_params['mode'] == 'dynamic':
        grid_params = create_dynamic_grid(fields_params, grid_params)
    
    x0, y0, z0 = grid_params['box_xyz']
    Nx, Ny, Nz = grid_params['Nxyz']
    x = np.linspace(-0.5*x0, 0.5*x0, Nx, endpoint=Nx%2)
    y = np.linspace(-0.5*y0, 0.5*y0, Ny, endpoint=Ny%2)
    z = np.linspace(-0.5*z0, 0.5*z0, Nz, endpoint=Nz%2)
    grid_xyz = GridXYZ((x, y, z))

    t0 = grid_params["box_t"]
    Nt = grid_params["Nt"]
    grid_t = np.linspace(-0.5*t0,0.5*t0,Nt)
    return grid_xyz, grid_t