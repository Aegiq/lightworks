"""
Lightworks
==========

Lightworks is a high-level interface for photonic quantum computing, 
enabling a range of transformations to be implemented and executed on both 
remote photonic hardware and with local simulation tools.

Key objects:

    Circuit : Provides a tool for building linear optic circuits across a 
        number of modes. Circuit also supports addition for combination of 
        sub-circuits and can be used with built-in Parameter objects to allow 
        for adjustment of the circuit configuration after creation. Circuit 
        also has a display method so the created circuit can be viewed.

    State : Represents the photonic fock states which are input and output from
        the system. State objects are hashable, and so can be used as keys in 
        dictionaries, and support both addition and merging to combine states
        together.

    emulator : Provides a set of local simulation tools for the testing and 
        verification of outputs from a given problem. There is a number of
        different objects for simulation of the system, which provide various
        capabilities and outputs.

"""

from .__version import __version__

from .sdk import Circuit, Unitary
from .sdk import Parameter, ParameterDict
from .sdk import Display
from .sdk import State
from .sdk import RemoteQPU, ResultProcessor
from .sdk import random_unitary, random_permutation
from .sdk import db_loss_to_transmission, transmission_to_db_loss
from .sdk import Optimisation
from .sdk import PresetCircuits
from .sdk import RemoteQPU

__all__ = ["Circuit", "Unitary", "Display", "State", "RemoteQPU", 
           "ResultProcessor", "random_unitary", "random_permutation",
           "db_loss_to_transmission", "transmission_to_db_loss",
           "Parameter", "ParameterDict", "Optimisation", "PresetCircuits"]