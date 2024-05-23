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
Contains a variety of qubit components, designed for implementing required 
qubit processing functionality in lightworks.
"""

from ...sdk.circuit import Unitary

import numpy as np

class H(Unitary):
    """
    Implements a Hadamard across a pair of modes corresponding to a dual-rail 
    encoded qubit.
    """
    def __init__(self) -> None:
        
        unitary = np.array([[1,1],[1,-1]])/2**0.5
        super().__init__(unitary, "H")


class X(Unitary):
    """
    Implements an X gate across a pair of modes corresponding to a dual-rail 
    encoded qubit.
    """
    def __init__(self) -> None:
        
        unitary = np.array([[0,1],[1,0]])
        super().__init__(unitary, "X")


class Y(Unitary):
    """
    Implements a Y gate across a pair of modes corresponding to a dual-rail 
    encoded qubit.
    """
    def __init__(self) -> None:
        
        unitary = np.array([[0,-1j],[1j,0]])
        super().__init__(unitary, "Y")


class Z(Unitary):
    """
    Implements a Z gate across a pair of modes corresponding to a dual-rail 
    encoded qubit.
    """
    def __init__(self) -> None:
        
        unitary = np.array([[1,0],[0,-1]])
        super().__init__(unitary, "Z")


class S(Unitary):
    """
    Implements an S gate across a pair of modes corresponding to a dual-rail 
    encoded qubit.
    """
    def __init__(self) -> None:
        
        unitary = np.array([[1,0],[0,1j]])
        super().__init__(unitary, "S")


class T(Unitary):
    """
    Implements a T gate across a pair of modes corresponding to a dual-rail 
    encoded qubit.
    """
    def __init__(self) -> None:
        
        unitary = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
        super().__init__(unitary, "T")