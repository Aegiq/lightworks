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
from ...sdk.circuit import Unitary, Circuit
from ...sdk.utils import permutation_mat_from_swaps_dict

import numpy as np

class CZ(Circuit):
    """
    Post-selected CZ gate that acts across two dual-rail encoded qubits. This 
    gate occupies a total of 6 modes, where modes 0 & 5 are used for 0 photon
    heralds, modes 1 & 2 correspond to the 0 & 1 states of the control qubit 
    and modes 3 & 4 correspond to the 0 & 1 states of the target qubit. This
    gate requires additional post-selection in which only one photon should be 
    measured across each of the pairs of modes which encode a qubit.    
    """
    def __init__(self) -> None:
            
        U_bs = np.array([[-1,2**0.5],[2**0.5,1]])/3**0.5
        U = np.identity(6, dtype = complex)
        for i in range(0,6,2):
            U[i:i+2,i:i+2] = U_bs[:,:]
        U[3,:] = -U[3,:]
        unitary = Unitary(U, label = "CZ")
        unitary.add_herald(0, 0, 0)
        unitary.add_herald(0, 5, 5)
        
        super().__init__(4)
        self.add(unitary, 0, group = True, name = "CZ")


class CNOT(Circuit):
    """
    Post-selected CNOT gate that acts across two dual-rail encoded qubits. This 
    gate occupies a total of 6 modes, where modes 0 & 5 are used for 0 photon
    heralds, modes 1 & 2 correspond to the 0 & 1 states of the control qubit 
    and modes 3 & 4 correspond to the 0 & 1 states of the target qubit. This
    gate requires additional post-selection in which only one photon should be 
    measured across each of the pairs of modes which encode a qubit.
    """
    def __init__(self) -> None:
        
        super().__init__(4)
        
        # Create CNOT from combination of H and CZ
        circ = Circuit(4)
        circ.add(H(), 2)
        circ.add(CZ(), 0)
        circ.add(H(), 2)
        
        self.add(circ, 0, group = True, name = "CNOT")

  
class CZ_Heralded(Circuit):
    """
    Heralded version of the CZ gate which acts across two dual-rail encoded 
    qubits, using two NS gates with ancillary photons to herald the success of 
    the transformation. This gate occupies 8 modes, where modes 0 & 7 are used 
    as 0 photon heralds, modes 1 & 6 are used as 1 photon heralds, mode 2 & 3
    correspond to the 0 & 1 states of the control qubit and modes 4 & 5 
    correspond to the 0 & 1 states of the target qubit. The heralded gate does
    not require any post-selection on the output qubits, other than that they
    are not lost (i.e a total of 4 photons should be measured at the output
    of the system), allowing it to be cascaded with other two qubit gates.
    """
    def __init__(self) -> None:
        
        U = np.identity(8, dtype = complex)
        
        U_ns = np.array([[1-2**0.5, 2**-0.25, (3/(2**0.5) - 2)**0.5],
                         [2**-0.25, 0.5, 0.5 - 2**-0.5],
                         [(3/(2**0.5) - 2)**0.5, 0.5 - 2**-0.5, 2**0.5 - 0.5]])
        U[1:4, 1:4] = np.flip(U_ns, axis=(0,1))[:,:]
        U[4:7, 4:7] = U_ns[:,:]
        # Apply pi phase shifts on mode 3
        U[:,3] = -U[:,3]
        
        # Define beam splitter action
        U_bs = np.identity(8, dtype=complex)
        U_bs[3,3] = 1/2**0.5
        U_bs[4,4] = 1/2**0.5
        U_bs[3,4] = 1j/2**0.5
        U_bs[4,3] = 1j/2**0.5
        
        # Define mode reconfiguration so qubits are on central 4 modes
        swaps = {2:0, 0:1, 1:2, 5:7, 7:6, 6:5}
        U_perm1 = permutation_mat_from_swaps_dict(swaps, 8)
        U_perm2 = np.conj(U_perm1.T)
        
        U = U_perm2 @ U_bs @ U @ U_bs @ U_perm1
        
        unitary = Unitary(U)
        unitary.add_herald(0, 0, 0)
        unitary.add_herald(1, 1, 1)
        unitary.add_herald(1, 6, 6)
        unitary.add_herald(0, 7, 7)
        
        super().__init__(4)
        self.add(unitary, 0, group = True, name = "CZ")


class CNOT_Heralded(Circuit):
    """
    Heralded version of the CNOT gate which acts across two dual-rail encoded 
    qubits, using two NS gates with ancillary photons to herald the success of 
    the transformation. This gate occupies 8 modes, where modes 0 & 7 are used 
    as 0 photon heralds, modes 1 & 6 are used as 1 photon heralds, mode 2 & 3
    correspond to the 0 & 1 states of the control qubit and modes 4 & 5 
    correspond to the 0 & 1 states of the target qubit. The heralded gate does
    not require any post-selection on the output qubits, other than that they
    are not lost (i.e a total of 4 photons should be measured at the output
    of the system), allowing it to be cascaded with other two qubit gates.
    """
    def __init__(self) -> None:
        
        super().__init__(4)
        
        # Create CNOT from combination of H and CZ
        circ = Circuit(4)
        circ.add(H(), 2)
        circ.add(CZ_Heralded(), 0)
        circ.add(H(), 2)
        
        self.add(circ, 0, group = True, name = "CNOT (Heralded)")