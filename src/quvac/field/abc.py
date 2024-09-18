'''
This script provides abstract interface for existing and future field
classes
'''

from abc import ABC, abstractmethod


class Field(ABC):
    @abstractmethod
    def calculate_field(self, t, **kwargs):
        '''
        Calculates fields for a given time step
        '''
        ...


class AnalyticField(Field):
    '''
    For such fields analytic formula is known for all time steps,
    every time step the formula is called to calculate the fields
    '''


class MaxwellField(Field):
    '''
    For such fields the initial field distribution (spectral coefficients)
    at a certain time step is given with analytic expression or from file.
    For later time steps the field is propagated according to linear Maxwell 
    equations 
    '''


class FieldFromFile(Field):
    '''
    (???) Potentially for fields that are pre-calculated somewhere else and
    are loaded from file
    '''


