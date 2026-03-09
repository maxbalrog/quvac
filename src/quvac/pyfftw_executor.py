'''
Pyfftw executor that is used in all 3D FFTs.

At the moment, batched FFT transforms of 3 3D arrays are performed.
'''
import os

import pyfftw

from quvac import config


class FFTExecutor:
    def __init__(self, tmp_shape, nthreads=None, fft_axes=(1,2,3)):
        self.tmp_shape = tmp_shape
        self.nthreads = nthreads if nthreads else os.cpu_count()
        self.fft_axes = fft_axes

    def allocate_fft(self):
        if getattr(self, "tmp", None) is None:
            self._allocate_fft()

    def _allocate_fft(self):
        self.tmp = pyfftw.zeros_aligned(self.tmp_shape, dtype=config.CDTYPE)
        
        self.forward_fftw = pyfftw.FFTW(
            self.tmp,
            self.tmp,
            axes=self.fft_axes,
            direction="FFTW_FORWARD",
            flags=(config.FFTW_FLAG,),
            threads=self.nthreads,
        )

        self.backward_fftw = pyfftw.FFTW(
            self.tmp,
            self.tmp,
            axes=self.fft_axes,
            direction="FFTW_BACKWARD",
            flags=(config.FFTW_FLAG,),
            threads=self.nthreads,
        )


def setup_fftw_executor(fft_executor, grid_shape, nthreads=None):
    if fft_executor is None:
        fft_executor = FFTExecutor(grid_shape, nthreads)
    fft_executor.allocate_fft()
    return fft_executor
