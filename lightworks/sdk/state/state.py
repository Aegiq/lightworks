# Copyright 2024 Aegiq Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A custom state datatype, which is created with the aim of making states in the
emulator much easier to work with.
"""

from ..utils import state_to_string
from ..utils import StateError

from copy import copy
from typing import Any

class State:
    """
    Custom data type to store information about a quantum state, as well as 
    allowing a number of operations to act on the state.
    
    Args:
    
        state (list) : The fock basis state to use with the class, this should
            be a list of photon numbers per mode.
    
    """

    def __init__(self, state: list) -> None:
        # If already list then assign to attribute
        if isinstance(state, list):
            self.__s = state
        # Otherwise try to convert
        else:
            self.__s = list(state)
        # Pre-calculate number of modes
        self.__n_modes = len(self.__s)
        return
    
    @property
    def n_photons(self) -> int:
        """Returns the number of photons in a State."""
        return sum(self.__s)
        
    @property
    def s(self) -> list:
        """Returns a copy of the contents of the state as a list."""
        return copy(self.__s)
    
    @s.setter
    def s(self, value: Any) -> None:
        raise StateError("State value should not be modified directly.")
    
    @property
    def n_modes(self) -> int:
        """The total number of modes in the state."""
        return self.__n_modes
    
    @n_modes.setter
    def n_modes(self, value: Any) -> None:
        raise StateError("Number of modes cannot be modified.")
    
    def merge(self, merge_state: "State") -> "State":
        """Combine two states, summing the number of photons per mode."""
        if self.__n_modes == merge_state.n_modes:
            return State([n1 + n2 for n1, n2 in zip(self.__s, merge_state.s)])
        else:
            raise ValueError("Merged states must be the same length.")

    def __str__(self) -> str:
        return state_to_string(self.__s)
    
    def __repr__(self) -> str: # Confirm this is the correct way to do this
            return f"State({state_to_string(self.__s)})"
    
    def __add__ (self, value: "State") -> "State":
        if isinstance(value, State):
            return State(self.__s + value.__s)
        else:
            raise TypeError("Addition only supported between states.")
        
    def __eq__(self, value: "State") -> bool:
        if isinstance(value, State):
            return self.__s == value.s
        else:
            return False
    
    def __hash__(self) -> str:
        return hash(self.__str__())
        
    def __len__(self) -> int:
        return self.__n_modes
    
    def __setitem__(self, key: Any, value: Any) -> None:
        raise StateError("State object does not support item assignment.")
    
    def __getitem__(self, indices: slice | int) -> "State":
        if isinstance(indices, slice):
            return State(self.__s[indices])
        elif isinstance(indices, int):
            return self.__s[indices]
        else:
            raise TypeError("Subscript should either be int or slice.")
        