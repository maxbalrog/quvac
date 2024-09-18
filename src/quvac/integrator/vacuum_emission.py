'''
This script implements calculation of vacuum emission integral
It is planned to add support for two versions:
    - Calculation of total vacuum emission signal for given field configuration
    - Separation of fields into pump and probe with subsequent calculation of probe channel signal
'''
'''
TODO:
    - Need a test for two colliding paraxial gaussians
    - Add calculation of total and polarization signal
'''

import numpy as np
import numexpr as ne
from scipy.constants import pi, c, epsilon_0, alpha, m_e, hbar, e
import pyfftw


BS = m_e**2 * c**2 / (hbar * e) # Schwinger magnetic field


class VacuumEmission(object):
    '''
    Calculator of Vacuum Emission amplitude from given fields

    Field parameters
    ----------------
    field: quvac.Field
        External fields
    '''
    def __init__(self, field):
        self.field = field
        self.allocate_fields()
        self.allocate_result_arrays()
        self.allocate_fft()
        self.setup_k_grid()

        angles = "theta phi beta".split()

        # Define symbolic expressions to evaluate later
        self.F = F = "0.5 * (Bx**2 + By**2 + Bz**2 - Ex**2 - Ey**2 - Ez**2)"
        self.G = G ="-(Ex*Bx + Ey*By + Ez*Bz)"
        self.U1 = [f"4*E{ax}*{F} + 7*B{ax}*{G}" for ax in "xyz"]
        self.U2 = [f"4*B{ax}*{F} - 7*E{ax}*{G}" for ax in "xyz"]
        self.I_ij = {f"{i}{j}": f"e{i}x*U{j}_acc_x + e{i}y*U{j}_acc_y + e{i}z*U{j}_acc_z"
                     for i in range(1,3) for j in range(1,3)}
        for key,val in self.I_ij.items():
            self.__dict__[f"I_{key}_expr"] = val
        # self.I = "cos(beta_p)*(I_11 - I_22) + sin(beta_p)*(I_12 + I_21)"
    
    def allocate_fields(self):
        self.E_out = [np.zeros(self.field.grid_shape) for _ in range(3)]
        self.B_out = [np.zeros(self.field.grid_shape) for _ in range(3)]
        self.Ex, self.Ey, self.Ez = self.E_out
        self.Bx, self.By, self.Bz = self.B_out

    def allocate_result_arrays(self):
        self.U1_acc = [np.zeros(self.field.grid_shape, dtype='complex128') for _ in range(3)]
        self.U2_acc = [np.zeros(self.field.grid_shape, dtype='complex128') for _ in range(3)]
        self.U1_acc_x, self.U1_acc_y, self.U1_acc_z = self.U1_acc
        self.U2_acc_x, self.U2_acc_y, self.U2_acc_z = self.U2_acc

    def allocate_fft(self):
        self.tmp = [pyfftw.zeros_aligned(self.field.grid_shape,  dtype='complex128') for _ in range(3)]
        # Add number of threads
        self.tmp_fftw = [pyfftw.FFTW(a, a, axes=(0, 1, 2),
                                    direction='FFTW_FORWARD',
                                    flags=('FFTW_MEASURE', ),)
                        for a in self.tmp]
    
    def setup_k_grid(self):
        self.dV, self.dVk = 1, 1
        for i,ax in enumerate('xyz'):
            grid = self.field.grid[i]
            self.__dict__[f'd{ax}'] = step = grid[1] - grid[0]
            self.__dict__[f'k{ax}'] = k_ = 2*pi*np.fft.fftfreq(grid.size, step)
            kstep = k_[1] - k_[0]
            self.dVk *= kstep
        N = np.prod(self.field.grid_shape)
        self.V = V = (2.*np.pi)**3 / self.dVk
        self.dV = V/N
        self.kmeshgrid = np.meshgrid(self.kx, self.ky, self.kz, indexing='ij', sparse=True)
        k_dict = {f'k{ax}': self.kmeshgrid[i] for i,ax in enumerate('xyz')}
        kx, ky, kz = k_dict["kx"], k_dict["ky"], k_dict["kz"]
        self.kabs = kabs = ne.evaluate("sqrt(kx**2 + ky**2 + kz**2)", local_dict=k_dict)
        kperp = ne.evaluate("sqrt(kx**2 + ky**2)", local_dict=k_dict)
        for i,ax in enumerate('xyz'):
            self.__dict__[f'k{ax}_unit'] = self.kmeshgrid[i] / self.kabs
            self.__dict__[f'k{ax}_unit'][0,0,0] = 0.
        self.e2x = ne.evaluate("where((kx==0) & (ky==0), 0.0, -ky / kperp)")
        self.e2y = ne.evaluate("where((kx==0) & (ky==0), 1.0, kx / kperp)")
        self.e2z = 0.

        self.e1x = ne.evaluate("where((kx==0) & (ky==0), 1.0, kx * kz / (kperp*kabs))")
        self.e1y = ne.evaluate("where((kx==0) & (ky==0), 0.0, ky * kz / (kperp*kabs))")
        self.e1z = ne.evaluate("where((kx==0) & (ky==0), 0.0, -kperp / kabs)")

    def free_resources(self):
        del self.E_out, self.B_out
        del self.tmp, self.tmp_fftw

    def calculate_one_time_step(self, t, weight=1):
        # Calculate fields
        self.allocate_fields()
        self.field.calculate_field(t, E_out=self.E_out, B_out=self.B_out)
        # Evaluate U1 and U2 expressions
        ax = 'xyz'
        for idx,U_expr in enumerate([self.U1, self.U2]):
            for i,expr in enumerate(U_expr):
                ne.evaluate(expr, global_dict=self.__dict__, out=self.tmp[i])
                self.tmp_fftw[i].execute()
                self.U = self.tmp[i]
                omega = self.kabs*c
                ne.evaluate(f"U{idx+1}_acc_{ax[i]} + U*exp(1j*omega*t)*dt*weight*dV",
                            global_dict=self.__dict__, out=self.__dict__[f"U{idx+1}_acc_{ax[i]}"])

    def calculate_time_integral(self, t_grid, integration_method="trapezoid"):
        self.dt = t_grid[1] - t_grid[0]
        if integration_method == "trapezoid":
            end_pts = (0,len(t_grid)-1)
            for i,t in enumerate(t_grid):
                weight = 0.5 if i in end_pts else 1.
                self.calculate_one_time_step(t, weight=weight)
        else:
            raise NotImplementedError(f"""integration_method should be one of ['trapezoid'] but 
                                      you passed {integration_method}""")

    def calculate_vacuum_current(self, t_grid, integration_method="trapezoid",
                                 filename="current.npz"):
        self.calculate_time_integral(t_grid, integration_method)
        self.free_resources()
        # Results should be in U1_acc and U2_acc
        self.S1 = ne.evaluate(f"{self.I_11_expr} - {self.I_22_expr}", global_dict=self.__dict__)
        self.S2 = ne.evaluate(f"{self.I_12_expr} + {self.I_21_expr}", global_dict=self.__dict__)

    def calculate_total_signal(self):
        # Calculate total signal
        S = ne.evaluate("S1.real**2 + S1.imag**2 + S2.real**2 + S2.imag**2",
                        global_dict=self.__dict__)
        prefactor = np.sqrt(alpha*self.kabs) / (2*pi)**1.5 / 45 / BS**3 * m_e**2 * c**3 / hbar**2
        Ntot = ne.evaluate("sum(prefactor**2 * S * dVk/(2*pi)**3)",
                           global_dict=globals() | self.__dict__)
        return Ntot


        
        




