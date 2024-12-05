"""
This script provides abstract interface for existing and future 
integrator classes
"""

from abc import ABC, abstractmethod


class Integrator(ABC):
    @abstractmethod
    def calculate_amplitudes(self, t_grid):
        """ """
        ...


class FourierIntegrator(Integrator):
    def allocate_fields(self):
        pass

    def calculate_one_time_step(self):
        pass
