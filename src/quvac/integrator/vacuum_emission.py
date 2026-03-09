"""
Calculation of vacuum emission integral (box diagram, F^4).

Currently supports:

1. Calculation of total vacuum emission signal for given field configuration. 
All fields are treated as external.

2. Separation of fields into pump and probe with subsequent calculation of
probe channel signal.

.. note::
    For details on implementation check out :ref:`implementation` section.
"""

import os
from pathlib import Path
import time

import numexpr as ne
import numpy as np
from scipy.constants import alpha, c, e, hbar, m_e, pi

from quvac import config
from quvac.pyfftw_executor import setup_fftw_executor

BS = m_e**2 * c**2 / (hbar * e)  # Schwinger magnetic field


class VacuumEmission:
    """
    Calculator of Vacuum Emission amplitude from given fields

    Parameters
    ----------
    field : quvac.Field
        External fields.
    grid : quvac.grid.GridXYZ
        Spatial and spectral grid.
    fft_executor: quvac.pyfftw_executor.FFTExecutor, optional
        Executor that performs FFTs.
    nthreads : int, optional
        Number of threads to use for calculations. If not provided, 
        defaults to the number of CPU cores.
    channels : bool, optional
        Whether to calculate a particular channel in vacuum emission amplitude. 
        Default is False.

    """

    def __init__(self, field, grid, fft_executor=None, nthreads=None, channels=False):
        self.field = field
        self.grid_xyz = grid
        # Update local dict with variables from GridXYZ class
        self.__dict__.update(self.grid_xyz.__dict__)
        self.channels = channels

        self.c = c
        self.nthreads = nthreads

        self.fft_executor = fft_executor

        # Define symbolic expressions to evaluate later
        self.F_expr = "(Bx**2 + By**2 + Bz**2 - Ex**2 - Ey**2 - Ez**2)/2"
        self.G_expr = "-(Ex*Bx + Ey*By + Ez*Bz)"

        self.F, self.G = [
            np.zeros(self.grid_shape, dtype=config.FDTYPE) for _ in range(2)
        ]

        if not self.channels:
            self.U1 = "(4*E*F + 7*B*G)"
            self.U2 = "(4*B*F - 7*E*G)"
        else:
            self._define_channel_variables()

        self.I_ij = {
            f"{i}{j}": f"(e{i}x*U{j}_acc_x + e{i}y*U{j}_acc_y + e{i}z*U{j}_acc_z)"
            for i in range(1, 3)
            for j in range(1, 3)
        }
        for key, val in self.I_ij.items():
            setattr(self, f"I_{key}_expr", val)

    def _define_channel_variables(self):
        """
        Define variables for channel-separated signal (linear in probe).
        """
        self.F_B_Bp_expr = "(Bx*Bpx + By*Bpy + Bz*Bpz - Ex*Epx - Ey*Epy - Ez*Epz)"
        self.G_Ep_B_expr = "-(Epx*Bx + Epy*By + Epz*Bz)"
        self.G_E_Bp_expr = "-(Ex*Bpx + Ey*Bpy + Ez*Bpz)"

        self.F_B_Bp, self.G_Ep_B, self.G_E_Bp = [
            np.zeros(self.grid_shape, dtype=config.FDTYPE) for _ in range(3)
        ]

        self.U1 = "(4*(Ep*F + E*F_B_Bp) + 7*(Bp*G + B*(G_Ep_B + G_E_Bp)))"
        self.U2 = "(4*(Bp*F + B*F_B_Bp) - 7*(Ep*G + E*(G_Ep_B + G_E_Bp)))"

    def _allocate_fields(self):
        """
        Allocate memory for field calculations.
        """
        self.E_out = np.zeros(self.vector_shape, dtype=config.CDTYPE)
        self.B_out = np.zeros(self.vector_shape, dtype=config.CDTYPE)
        if self.channels:
            self.E_probe = np.zeros(self.vector_shape, dtype=config.CDTYPE)
            self.B_probe = np.zeros(self.vector_shape, dtype=config.CDTYPE)

    def _allocate_result_arrays(self):
        """
        Allocate memory for result arrays.
        """
        self.U1_acc = np.zeros(self.vector_shape, dtype=config.CDTYPE)
        self.U2_acc = np.zeros(self.vector_shape, dtype=config.CDTYPE)
        self.U1_acc_x, self.U1_acc_y, self.U1_acc_z = self.U1_acc
        self.U2_acc_x, self.U2_acc_y, self.U2_acc_z = self.U2_acc

        self.U_pairs = [
            (self.U1_acc, self.U1),
            (self.U2_acc, self.U2),
        ]

        self.prefactor = np.ones(self.grid_shape, dtype="complex128")
        self.prefactor_step = np.zeros(self.grid_shape, dtype="complex128")

        self.U_dict = {"F": self.F, "G": self.G}

    def _allocate_fft(self):
        """
        Allocate memory for FFT calculations.
        """
        self.fft_executor = setup_fftw_executor(self.fft_executor, self.vector_shape, 
                                                self.nthreads)

        self.U_acc_dict = {
            "U": self.fft_executor.tmp,
            "prefactor": self.prefactor,
        }

    def _allocate_resources(self):
        """
        Allocate arrays needed for calculation.
        """
        self._allocate_result_arrays()
        self._allocate_fft()
        self._allocate_fields()

    def _free_resources(self):
        """
        Free allocated resources.
        """
        del self.E_out, self.B_out
        del self.fft_executor

    def calculate_one_time_step(self, t, weight=1):
        """
        Calculate the field and U terms (integrand) for one time step.
        """
        # Calculate fields
        if not self.channels:
            self.field.calculate_field(t, E_out=self.E_out, B_out=self.B_out)
        else:
            self.field.calculate_field(
                t,
                E_probe=self.E_probe,
                B_probe=self.B_probe,
                E_pump=self.E_out,
                B_pump=self.B_out,
            )
            Ep = Epx, Epy, Epz = np.real(self.E_probe)
            Bp = Bpx, Bpy, Bpz = np.real(self.B_probe)

        E = Ex, Ey, Ez = np.real(self.E_out)
        B = Bx, By, Bz = np.real(self.B_out)
        self.U_dict.update({"E": E, "B": B,
                            "Ex": Ex, "Ey": Ey, "Ez": Ez, 
                            "Bx": Bx, "By": By, "Bz": Bz,})

        # Evaluate F and G
        ne.evaluate(self.F_expr, out=self.F)
        ne.evaluate(self.G_expr, out=self.G)

        if self.channels:
            ne.evaluate(self.F_B_Bp_expr, out=self.F_B_Bp)
            ne.evaluate(self.G_Ep_B_expr, out=self.G_Ep_B)
            ne.evaluate(self.G_E_Bp_expr, out=self.G_E_Bp)
            self.U_dict.update({"Ep": Ep, "Bp": Bp,
                                "Epx": Epx, "Epy": Epy, "Epz": Epz,
                                "Bpx": Bpx, "Bpy": Bpy, "Bpz": Bpz,
                                "F_B_Bp": self.F_B_Bp,
                                "G_Ep_B": self.G_Ep_B,
                                "G_E_Bp": self.G_E_Bp,})

        # Update prefactor: this prescription is valid only for equidistant grids!!!
        ne.evaluate("prefactor*prefactor_step", local_dict=self.prefactor_dict, 
                    out=self.prefactor)

        # Evaluate U1 and U2 expressions
        for U_acc, U_expr in self.U_pairs: # noqa: B905
            ne.evaluate(U_expr, global_dict=self.U_dict, out=self.fft_executor.tmp)
            self.fft_executor.forward_fftw.execute()

            self.U_acc_dict.update({"U_acc": U_acc})
            ne.evaluate(
                "U_acc + U*prefactor",
                local_dict=self.U_acc_dict,
                out=U_acc,
            )

    def multiply_integration_result(self, t_grid):
        """
        Multiply the integral by common prefactors.
        """
        self._free_resources()
        # prefactor related to time grid and discretization
        self.prefactor_dict.update({"t": t_grid[0], "dt": self.dt, "dV": self.dV,})
        ne.evaluate(
            "exp(1j*kabs*c*t)*dt*dV", 
            local_dict=self.prefactor_dict, 
            out=self.prefactor
        )
        for acc in [self.U1_acc, self.U2_acc]:
            ne.evaluate("acc*prefactor", global_dict={"prefactor": self.prefactor},
                        out=acc)

    def calculate_time_integral(self, t_grid, integration_method="trapezoid"):
        """
        Calculate the time integral.
        """
        self.dt = t_grid[1] - t_grid[0]
        # prefactor-related variables
        self.prefactor_dict = {
            "kabs": self.kabs,
            "c": c,
            "dt": self.dt,
            "prefactor": self.prefactor, 
            "prefactor_step": self.prefactor_step,
        }
        ne.evaluate(
            "exp(1j*kabs*c*dt)", local_dict=self.prefactor_dict, out=self.prefactor_step
        )

        if integration_method == "trapezoid":
            # end_pts = (0, len(t_grid) - 1)
            for _, t in enumerate(t_grid):
                # weight = 0.5 if i in end_pts else 1.
                weight = 1
                self.calculate_one_time_step(t, weight=weight)
        else:
            err_msg = (
                "integration_method should be one of ['trapezoid'] but you "
                f"passed {integration_method}"
            )
            raise NotImplementedError(err_msg)
        
        # finish calculation
        self.multiply_integration_result(t_grid)

    def _calculate_S1_S2(self):
        """
        Calculate S1 and S2 amplitudes from accumulated values of integrals U1 and U2.
        """
        dims = 1 / BS**3 * m_e**2 * c**3 / hbar**2
        prefactor = -1j * np.sqrt(alpha * self.kabs) / (2 * pi) ** 1.5 / 45 * dims # noqa: F841
        self.S1 = ne.evaluate(
            f"prefactor * ({self.I_11_expr} - {self.I_22_expr})",
            global_dict=self.__dict__,
        ).astype(config.CDTYPE)
        self.S2 = ne.evaluate(
            f"prefactor * ({self.I_12_expr} + {self.I_21_expr})",
            global_dict=self.__dict__,
        ).astype(config.CDTYPE)

    def calculate_amplitudes(
        self, t_grid, integration_method="trapezoid", save_path=None
    ):
        """
        Calculate the vacuum emission amplitudes and save the result.
        """
        self._allocate_resources()

        time_integral_start = time.perf_counter()
        self.calculate_time_integral(t_grid, integration_method)
        time_integral_end = time.perf_counter()
        time_integral = time_integral_end - time_integral_start

        # Calculate 
        self._calculate_S1_S2()
        
        # Save amplitudes
        if save_path:
            self.save_amplitudes(save_path)
        return time_integral

    def save_amplitudes(self, save_path):
        """
        Save the calculated amplitudes to a file.
        """
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        data = {
            "x": self.grid[0],
            "y": self.grid[1],
            "z": self.grid[2],
            "S1": self.S1,
            "S2": self.S2,
        }
        np.savez(save_path, **data)
