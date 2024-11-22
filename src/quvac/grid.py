'''
This script implements:
1) GridXYZ class that calculates necessary variables
   for spatial grid and its Fourier counterpart;
2) helper function `setup_grids` that automatically 
   creates grids either dynamically or from given grid params

Dynamical grid creation (`get_kmax_grid`, `get_xyz_size`,
`get_t_size` were originally implemented by Alexander Blinne)
'''

from collections.abc import Iterable

import numpy as np
import numexpr as ne
from scipy.constants import pi, c
import pyfftw


class GridXYZ(object):
    '''
    Calculates space grid and its Fourier counterpart

    Parameters:
    -----------
    grid: (np.array, np.array, np.array) = (x, y, z)
        Spatial grid for field discretization
    '''
    def __init__(self, grid):
        # Define spatial grid
        self.grid = grid
        self.grid_shape = tuple(ax.size for ax in grid)
        self.N = np.prod(self.grid_shape)
        self.xyz = self.x, self.y, self.z = np.meshgrid(*grid, indexing='ij', sparse=True)
        self.dxyz = tuple(ax[1]-ax[0] for ax in grid)
        self.dV = np.prod(self.dxyz)

    def get_k_grid(self):
        for i,ax in enumerate('xyz'):
            self.__dict__[f'k{ax}'] = 2.*pi*np.fft.fftfreq(self.grid_shape[i],
                                                               self.dxyz[i]) 
        self.kgrid = tuple((self.kx, self.ky, self.kz))
        self.kgrid_shifted = tuple(np.fft.fftshift(k) for k in (self.kx, self.ky, self.kz))
        self.dkxkykz = [ax[1]-ax[0] for ax in self.kgrid]
        self.dVk = np.prod(self.dkxkykz)

        self.kmeshgrid = np.meshgrid(self.kx, self.ky, self.kz, indexing='ij', sparse=True)
        kx, ky, kz = self.kmeshgrid
        self.kabs = kabs = ne.evaluate("sqrt(kx**2 + ky**2 + kz**2)")
        kperp = ne.evaluate("sqrt(kx**2 + ky**2)")

        # Polarization vectors
        self.e1x = ne.evaluate("where((kx==0) & (ky==0), 1.0, kx * kz / (kperp*kabs))")
        self.e1y = ne.evaluate("where((kx==0) & (ky==0), 0.0, ky * kz / (kperp*kabs))")
        self.e1z = ne.evaluate("where((kx==0) & (ky==0), 0.0, -kperp / kabs)")

        self.e2x = ne.evaluate("where((kx==0) & (ky==0), 0.0, -ky / kperp)")
        self.e2y = ne.evaluate("where((kx==0) & (ky==0), 2*(kz>0)-1, kx / kperp)")
        self.e2z = 0.


def get_ek(theta, phi):
    '''
    Calculate k-vector for given spherical angles
    theta and phi
    '''
    ek = np.array([np.sin(theta) * np.cos(phi),
                   np.sin(theta) * np.sin(phi),
                   np.cos(theta)])
    return ek


def get_pol_basis(theta, phi):
    '''
    Calculate polarization basis for given spherical angles
    theta and phi
    '''
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

    Parameters:
    -----------
    field_params: dict
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
    
    Parameters:
    -----------
    fields: dict | list of dicts
        Parameters of participating fields
    box_size: typing.Sequence (e.g., list, np.array)
        Box size for 3 dimensions
    grid_res: float
        Controls the resolution
    equal_resolution: bool
        Flag if equal resolution in every dimension
        is needed
    '''
    if isinstance(fields, dict):
        fields = list(fields.values())
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
    
    Parameters:
    -----------
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
    '''
    Dynamically create grids from given laser parameters.
    One should be careful with this option.

    Parameters:
    -----------
    fields_params: list of dicts
        Parameters of participating fields
    grid_params: dict
        Grid parameters
        Required keys: transverse_factor, longitudinal_factor, time_factor,
        spatial_resolution, time_resolution
    '''
    # Create spatial box
    collision_geometry = grid_params.get('collision_geometry', 'z')

    w0_max = max([field.get('w0', 0) for field in fields_params])
    tau_max = max([field.get('tau', 0) for field in fields_params])
    lam_min = min([field.get('lam') for field in fields_params])
    
    transverse_size = w0_max * grid_params['transverse_factor']
    longitudinal_size = tau_max * c * grid_params['longitudinal_factor']

    box_xyz = [0, 0, 0]
    for i,ax in enumerate('xyz'):
        box_xyz[i] = (longitudinal_size if ax in collision_geometry 
                      else transverse_size)
    grid_params['box_xyz'] = box_xyz

    # Number of spatial pts 
    box_size = np.array(box_xyz) / 2
    Nxyz_ = get_xyz_size(fields_params, box_size)
    res = grid_params['spatial_resolution']
    if isinstance(res, Iterable):
        Nxyz = [Nx * res for Nx,res in zip(Nxyz_, res)]
    elif type(res) in (int, float):
        Nxyz = [Nx * res for Nx in Nxyz_]
    grid_params['Nxyz'] = Nxyz

    # Create temporal box
    t0 = tau_max * grid_params['time_factor']
    Nt = get_t_size(-t0/2, t0/2, lam_min)

    # Number of temporal pts
    grid_params['box_t'] = t0
    grid_params['Nt'] = Nt * grid_params['time_resolution']
    return grid_params
    

def setup_grids(fields_params, grid_params):
    '''
    Create spatial and temporal grids from given sizes or
    dynamically from field parameters (e.g., tau, w0)

    Parameters:
    -----------
    fields_params: list of dicts
        Parameters of participating fields
    grid_params: dict
        Grid parameters
        Required_keys: mode (other keys depend on mode)
    '''
    if grid_params['mode'] == 'dynamic':
        if isinstance(fields_params, dict):
            fields_params = list(fields_params.values())
        grid_params = create_dynamic_grid(fields_params, grid_params)
    
    x0, y0, z0 = grid_params['box_xyz']
    Nx, Ny, Nz = grid_params['Nxyz']
    x = np.linspace(-0.5*x0, 0.5*x0, Nx, endpoint=Nx%2)
    y = np.linspace(-0.5*y0, 0.5*y0, Ny, endpoint=Ny%2)
    z = np.linspace(-0.5*z0, 0.5*z0, Nz, endpoint=Nz%2)
    grid_xyz = GridXYZ((x, y, z))

    Nt = grid_params["Nt"]
    box_t = grid_params["box_t"]
    # Allow non-symmetric time-grids
    if isinstance(box_t, float):
        Nt += int(1 - Nt%2)
        t0 = box_t
        grid_t = np.linspace(-0.5*t0, 0.5*t0, Nt, endpoint=True)
    else:
        t_start, t_end = box_t
        grid_t = np.linspace(t_start, t_end, Nt, endpoint=True)
    return grid_xyz, grid_t