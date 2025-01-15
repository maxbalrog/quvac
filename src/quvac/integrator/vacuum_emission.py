"""
This script implements calculation of vacuum emission integral (box diagram, F^4)
It is planned to add support for two versions:
    - Calculation of total vacuum emission signal for given field configuration
    - Separation of fields into pump and probe with subsequent calculation of probe channel signal
"""

import os
import time
from pathlib import Path

import numexpr as ne
import numpy as np
import pyfftw
from scipy.constants import alpha, c, e, hbar, m_e, pi

from quvac import config

BS = m_e**2 * c**2 / (hbar * e)  # Schwinger magnetic field


class VacuumEmission(object):
    """
    Calculator of Vacuum Emission amplitude from given fields

    Parameters
    ----------
    field: quvac.Field
        External fields
    grid: quvac.grid.GridXYZ
        spatial and spectral grid
    channels: bool
        Whether to calculate a particular channel in vacuum emission
        amplitude
    """

    def __init__(self, field, grid, nthreads=None, channels=False):
        self.field = field
        self.grid_xyz = grid
        # Update local dict with variables from GridXYZ class
        self.__dict__.update(self.grid_xyz.__dict__)
        self.channels = channels

        self.c = c
        self.nthreads = nthreads if nthreads else os.cpu_count()

        # Define symbolic expressions to evaluate later
        self.F_expr = "(Bx**2 + By**2 + Bz**2 - Ex**2 - Ey**2 - Ez**2)/2"
        self.G_expr = "-(Ex*Bx + Ey*By + Ez*Bz)"

        self.F, self.G = [
            np.zeros(self.grid_shape, dtype=config.FDTYPE) for _ in range(2)
        ]

        if not self.channels:
            self.U1 = [f"(4*E{ax}*F + 7*B{ax}*G)" for ax in "xyz"]
            self.U2 = [f"(4*B{ax}*F - 7*E{ax}*G)" for ax in "xyz"]
        else:
            self.define_channel_variables()

        self.I_ij = {
            f"{i}{j}": f"(e{i}x*U{j}_acc_x + e{i}y*U{j}_acc_y + e{i}z*U{j}_acc_z)"
            for i in range(1, 3)
            for j in range(1, 3)
        }
        for key, val in self.I_ij.items():
            self.__dict__[f"I_{key}_expr"] = val

    def define_channel_variables(self):
        self.F_B_Bp_expr = "(Bx*Bpx + By*Bpy + Bz*Bpz - Ex*Epx - Ey*Epy - Ez*Epz)"
        self.G_Ep_B_expr = "-(Epx*Bx + Epy*By + Epz*Bz)"
        self.G_E_Bp_expr = "-(Ex*Bpx + Ey*Bpy + Ez*Bpz)"

        self.F_B_Bp, self.G_Ep_B, self.G_E_Bp = [
            np.zeros(self.grid_shape, dtype=config.FDTYPE) for _ in range(3)
        ]

        self.U1 = [
            f"(4*(Ep{ax}*F + E{ax}*F_B_Bp) + 7*(Bp{ax}*G + B{ax}*(G_Ep_B + G_E_Bp)))"
            for ax in "xyz"
        ]
        self.U2 = [
            f"(4*(Bp{ax}*F + B{ax}*F_B_Bp) - 7*(Ep{ax}*G + E{ax}*(G_Ep_B + G_E_Bp)))"
            for ax in "xyz"
        ]

    def allocate_fields(self):
        self.E_out = [np.zeros(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)]
        self.B_out = [np.zeros(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)]
        if self.channels:
            self.E_probe = [
                np.zeros(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)
            ]
            self.B_probe = [
                np.zeros(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)
            ]

    def allocate_result_arrays(self):
        self.U1_acc = [np.zeros(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)]
        self.U2_acc = [np.zeros(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)]
        self.U1_acc_x, self.U1_acc_y, self.U1_acc_z = self.U1_acc
        self.U2_acc_x, self.U2_acc_y, self.U2_acc_z = self.U2_acc

    def allocate_fft(self):
        self.tmp = [
            pyfftw.zeros_aligned(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)
        ]
        # Add number of threads
        self.tmp_fftw = [
            pyfftw.FFTW(
                a,
                a,
                axes=(0, 1, 2),
                direction="FFTW_FORWARD",
                flags=("FFTW_MEASURE",),
                threads=self.nthreads,
            )
            for a in self.tmp
        ]
        self.prefactor = np.zeros(self.grid_shape, dtype="complex128")

        self.prefactor_dict = {
            "kabs": self.kabs,
            "c": c,
        }

        self.U_dict = {"F": self.F, "G": self.G}

    def free_resources(self):
        del self.E_out, self.B_out
        del self.tmp, self.tmp_fftw

    def calculate_one_time_step(self, t, weight=1):
        # Calculate fields
        self.allocate_fields()

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
            Epx, Epy, Epz = [E.real for E in self.E_probe]
            Bpx, Bpy, Bpz = [B.real for B in self.B_probe]

        Ex, Ey, Ez = [E.real for E in self.E_out]
        Bx, By, Bz = [B.real for B in self.B_out]
        self.U_dict.update({"Ex": Ex, "Ey": Ey, "Ez": Ez, "Bx": Bx, "By": By, "Bz": Bz})

        ne.evaluate(self.F_expr, out=self.F)
        ne.evaluate(self.G_expr, out=self.G)

        if self.channels:
            ne.evaluate(self.F_B_Bp_expr, out=self.F_B_Bp)
            ne.evaluate(self.G_Ep_B_expr, out=self.G_Ep_B)
            ne.evaluate(self.G_E_Bp_expr, out=self.G_E_Bp)
            self.U_dict.update({"Epx": Epx, "Epy": Epy, "Epz": Epz,
                                "Bpx": Bpx, "Bpy": Bpy, "Bpz": Bpz,
                                "F_B_Bp": self.F_B_Bp,
                                "G_Ep_B": self.G_Ep_B,
                                "G_E_Bp": self.G_E_Bp,})

        # Evaluate U1 and U2 expressions
        self.prefactor_dict.update({"t": t})
        ne.evaluate(
            "exp(1j*kabs*c*t)", local_dict=self.prefactor_dict, out=self.prefactor
        )
        for U_key, U_expr in zip(["U1_acc", "U2_acc"], [self.U1, self.U2]):
            for i, expr in enumerate(U_expr):
                U_acc = getattr(self, U_key)[i]
                ne.evaluate(expr, global_dict=self.U_dict, out=self.tmp[i])
                self.tmp_fftw[i].execute()

                U_res = ne.evaluate(
                    f"U_acc + U*prefactor*dt*dV",
                    global_dict={
                        "U_acc": U_acc,
                        "U": self.tmp[i],
                        "prefactor": self.prefactor,
                        "dt": self.dt,
                        "dV": self.dV,
                    },
                )
                U_acc[:] = U_res.astype(config.CDTYPE)

    def calculate_time_integral(self, t_grid, integration_method="trapezoid"):
        self.dt = t_grid[1] - t_grid[0]
        if integration_method == "trapezoid":
            end_pts = (0, len(t_grid) - 1)
            for i, t in enumerate(t_grid):
                # weight = 0.5 if i in end_pts else 1.
                weight = 1
                self.calculate_one_time_step(t, weight=weight)
        else:
            err_msg = (
                "integration_method should be one of ['trapezoid'] but you "
                f"passed {integration_method}"
            )
            raise NotImplementedError(err_msg)

    def calculate_amplitudes(
        self, t_grid, integration_method="trapezoid", save_path=None
    ):
        # Allocate resources
        self.allocate_result_arrays()
        self.allocate_fft()

        time_integral_start = time.perf_counter()
        self.calculate_time_integral(t_grid, integration_method)
        time_integral_end = time.perf_counter()
        time_integral = time_integral_end - time_integral_start
        self.free_resources()

        # Results should be in U1_acc and U2_acc
        dims = 1 / BS**3 * m_e**2 * c**3 / hbar**2
        prefactor = -1j * np.sqrt(alpha * self.kabs) / (2 * pi) ** 1.5 / 45 * dims
        # Next time need to be careful with f-strings and brackets
        self.S1 = ne.evaluate(
            f"prefactor * ({self.I_11_expr} - {self.I_22_expr})",
            global_dict=self.__dict__,
        ).astype(config.CDTYPE)
        self.S2 = ne.evaluate(
            f"prefactor * ({self.I_12_expr} + {self.I_21_expr})",
            global_dict=self.__dict__,
        ).astype(config.CDTYPE)
        # Save amplitudes
        if save_path:
            self.save_amplitudes(save_path)
        return time_integral

    def save_amplitudes(self, save_path):
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        data = {
            "x": self.grid[0],
            "y": self.grid[1],
            "z": self.grid[2],
            "S1": self.S1,
            "S2": self.S2,
        }
        np.savez(save_path, **data)
