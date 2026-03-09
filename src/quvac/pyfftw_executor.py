'''
Pyfftw executor that is used in all 3D FFTs.

At the moment, batched FFT transforms of three 3D arrays are performed (corresponding
to a vector field on 3D grid).
'''
import os

import pyfftw

from quvac import config


class FFTExecutor:
    """
    Class to unite the FFT calculation.

    Parameters
    ----------
    tmp_shape : tuple or list
        Size of the buffer array over which FFTs are performed. By default, it expects 
        (3, Nx, Ny, Nz).
    nthreads : int, optional
        Number of threads to use for calculations. If not provided, defaults to the 
        number of CPU cores.
    fft_axes : tuple or list
        Axes over which to perform FFT. By default, (1,2,3).

    Attributes
    ----------
    tmp_shape : tuple
        Size of the buffer array.
    nthreads : int
        Number of threads to use for FFTs.
    fft_axes : tuple
        Axes over which to perform FFT.
    tmp : np.ndarray of size `tmp_shape`
        Buffer array used for FFTs.
    forward_fftw : pyfftw.FFTW
        Forward FFT executor.
    backward_fftw : pyfftw.FFTW
        Backward FFT executor.
    """
    def __init__(self, tmp_shape, nthreads=None, fft_axes=(1,2,3)):
        self.tmp_shape = tuple(tmp_shape)
        self.nthreads = nthreads if nthreads else os.cpu_count()
        self.fft_axes = tuple(fft_axes)

    def allocate_fft(self):
        """
        Allocate buffer array and set up FFT executors if they were not already.
        """
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
    """
    Unified function to:
        - (Optional )Set up FFTExecutor if it was not already.
        - Allocate buffer arrays and FFT executors.
    """
    if fft_executor is None:
        fft_executor = FFTExecutor(grid_shape, nthreads)
    fft_executor.allocate_fft()
    return fft_executor
