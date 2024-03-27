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
A custom state datatype, which is created for storing annotated state details.
It is not intended that this class will be ordinarily accessible to users.
"""

from ..utils import annotated_state_to_string
from ..utils import AnnotatedStateError

from typing import Any

class AnnotatedState:
    """
    Acts as a custom data state which enables fock states to be defined, with 
    the main difference to the core State class being that here they are 
    defined with a label to indicate photon indistinguishability. 

    Args:
    
        state (list) : The fock basis state to use with the class, this should
            be a list of lists, each containing labels for the photons. The 
            number of labels in the list will dictate the total photon number 
            in the mode.
    
    """
    
    __slots__ = ["__s"]
    def __init__(self, state: list) -> None:
        for s in state:
            if type(s) != list:
                raise TypeError("Provided state labels should be lists.")
        self.__s = [sorted(s) for s in state]
        return
    
    @property
    def n_photons(self) -> int:
        """Returns the number of photons in a State."""
        return sum([len(s) for s in self.__s])
    
    @property
    def s(self) -> None:
        return [[j for j in i] for i in self.__s]
    
    @s.setter
    def s(self, value: Any) -> None:
        raise AnnotatedStateError(
            "State value should not be modified directly.")
    
    @property
    def n_modes(self) -> None:
        return len(self.__s)
    
    @n_modes.setter
    def n_modes(self, value: Any) -> None:
        raise AnnotatedStateError("Number of modes cannot be modified.")
    
    def merge(self, merge_state: "AnnotatedState") -> "AnnotatedState":
        """Combine two states, summing the number of photons per mode."""
        if self.n_modes == merge_state.n_modes:
            return AnnotatedState([n1 + n2 for n1, n2 in zip(self.__s, 
                                                             merge_state.s)])
        else:
            raise ValueError("Merged states must be the same length.")
    
    def __str__(self) -> str:
        return annotated_state_to_string(self.__s)
    
    def __repr__(self) -> str:
            return f"AnnotatedState({annotated_state_to_string(self.__s)})"
    
    def __add__ (self, value: "AnnotatedState") -> "AnnotatedState":
        if isinstance(value, AnnotatedState):
            return AnnotatedState(self.__s + value.__s)
        else:
            raise TypeError(
                "Addition only supported between annotated states.")
        
    def __eq__(self, value: "AnnotatedState") -> bool:
        if isinstance(value, AnnotatedState):
            return self.__s == value.__s
        else:
            return False
    
    def __hash__(self) -> str:
        return hash(self.__str__())
        
    def __len__(self) -> int:
        return self.n_modes
    
    def __setitem__(self, key: Any, value: Any) -> None:
        raise AnnotatedStateError(
            "AnnotatedState object does not support item assignment.")
    
    def __getitem__(self, indices: int | slice) -> "AnnotatedState":
        if isinstance(indices, slice):
            return AnnotatedState(self.__s[indices])
        elif isinstance(indices, int):
            return self.__s[indices]
        else:
            raise TypeError("Subscript should either be int or slice.")