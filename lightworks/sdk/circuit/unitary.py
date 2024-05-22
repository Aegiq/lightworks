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
Dedicated unitary component for implementing unitary matrices on a circuit.
"""

from .circuit import Circuit
from ..utils import check_unitary

import numpy as np

class Unitary(Circuit):
    """
    Create a circuit which implements the target provided unitary across all of
    its modes.
    
    Args:
    
        unitary (np.ndarray) : The target NxN unitary matrix which is to be 
            implemented.
    
    """
    def __init__(self, unitary: np.ndarray, label: str = "U") -> None:
        
        # Check type of supplied unitary
        if not isinstance(unitary, (np.ndarray, list)):
            raise TypeError("Unitary should be a numpy array or list.")
        unitary = np.array(unitary)
        
        # Ensure unitary is valid
        if not check_unitary(unitary):
            raise ValueError("Provided matrix is not unitary.")
        
        # Also check label is a string
        if not isinstance(label, str):
            raise TypeError("Label for unitary should be a string.")
        
        super().__init__(int(unitary.shape[0]))
        self._Circuit__circuit_spec = [["unitary", (0, unitary, label)]]
        
        return