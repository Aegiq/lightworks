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
Contains a variety of two qubit components, designed for implementing required 
qubit processing functionality in lightworks.
"""

from .single_qubit_gates import H
from ..circuit import Unitary, Circuit

import numpy as np

class CZ(Unitary):
    """
    Post-selected CZ gate that acts across two dual-rail encoded qubits. This 
    gate occupies a total of 6 modes, where modes 0 & 5 are used for 0 photon
    heralds, modes 1 & 2 correspond to the 0 & 1 states of the control qubit 
    and modes 3 & 4 correspond to the 0 & 1 states of the target qubit. This
    date requires additional post-selection in which only one photon should be 
    measured across each of the pairs of modes which encode a qubit.
          ______
     0 --|      | -- 0
    c0 --|      | -- \ 1 photon
    c1 --|  CZ  | -- /
    t0 --|      | -- \ 1 photon
    t1 --|      | -- /
     0 --|______| -- 0 
    
    """
    def __init__(self) -> None:
            
        u_bs = np.array([[-1,2**0.5],[2**0.5,1]])/3**0.5
        unitary = np.identity(6, dtype = complex)
        for i in range(0,6,2):
            unitary[i:i+2,i:i+2] = u_bs[:,:]
        unitary[3,:] = -unitary[3,:]
        
        super().__init__(unitary, "CZ")
        
class CNOT(Circuit):
    """
    Post-selected CNOT gate that acts across two dual-rail encoded qubits. This 
    gate occupies a total of 6 modes, where modes 0 & 5 are used for 0 photon
    heralds, modes 1 & 2 correspond to the 0 & 1 states of the control qubit 
    and modes 3 & 4 correspond to the 0 & 1 states of the target qubit. This
    date requires additional post-selection in which only one photon should be 
    measured across each of the pairs of modes which encode a qubit.
          ________
     0 --|        | -- 0
    c0 --|        | -- \ 1 photon
    c1 --|  CNOT  | -- /
    t0 --|        | -- \ 1 photon
    t1 --|        | -- /
     0 --|________| -- 0 
    
    """
    def __init__(self) -> None:
        
        super().__init__(6)
        
        # Create CZ from combination of H and CZ
        circ = Circuit(6)
        circ.add(H(), 3)
        circ.add(CZ(), 0)
        circ.add(H(), 3)
        
        self.add(circ, 0, group = True, name = "CNOT")