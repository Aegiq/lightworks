"""Dedicated unitary component for implement unitary matrices on a circuit."""

from .circuit import Circuit
from ..utils import check_unitary

import numpy as np

class Unitary(Circuit):
    """
    Unitary class
    This class can be used to create a circuit which implements the target 
    provided unitary across all of its modes.
    
    Args:
    
        unitary (np.ndarray) : The target NxN unitary matrix which is to be 
            implemented.
    
    """
    def __init__(self, unitary: np.ndarray) -> None:
        
        # Check type of supplied unitary
        if not isinstance(unitary, (np.ndarray, list)):
            raise TypeError("Unitary should be a numpy array or list.")
        unitary = np.array(unitary)
        
        # Ensure unitary is valid
        if not check_unitary(unitary):
            raise ValueError("Matrix is not unitary.")
        
        super().__init__(int(unitary.shape[0]))
        self._Circuit__circuit_spec = [["unitary", (0, unitary)]]
        
        return