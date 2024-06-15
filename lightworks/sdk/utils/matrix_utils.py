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
This file contains a collection of different useful functions for operations on
matrices.
"""

import numpy as np
from scipy.stats import unitary_group
from typing import Any

def random_unitary(N: int, seed: int | None = None) -> np.ndarray:
    """
    Generate a random NxN unitary matrix. Seed can be used to produce the same
    unitary each time the function is called.
    
    Args:
    
        N (int) : The dimension of the random unitary that is to be generated.
        
        seed (int | None, optional) : Specify a random seed to repeatedly 
            produce the same unitary matrix on each function call. Defaults to 
            None, which will produce a random matrix on each call.
            
    Returns:
    
        np.ndarray : The created random unitary matrix. 
        
    Raises:
    
        TypeError : In cases where the supplied random seed is not an integer
            value. 
    
    """
    seed = _check_random_seed(seed)
    return unitary_group.rvs(N, random_state = seed)

def random_permutation(N: int, seed: int | None = None) -> np.ndarray:
    """
    Generate a random NxN permutation. Seed can be used to produce the same
    unitary each time the function is called.
    
    Args:
    
        N (int) : The dimension of the random permutation that is to be 
            generated.
        
        seed (int | None, optional) : Specify a random seed to repeatedly 
            produce the same unitary matrix on each function call. Defaults to 
            None, which will produce a random matrix on each call.
            
    Returns:
    
        np.ndarray : The created random permutation matrix. 
        
    Raises:
    
        TypeError : In cases where the supplied random seed is not an integer
            value. 
            
    """
    seed = _check_random_seed(seed)
    np.random.seed(seed)
    return np.random.permutation(np.identity(N, dtype = complex))

def _check_random_seed(seed: Any) -> int | None:
    """Process a supplied random seed."""
    if not isinstance(seed, (int, type(None))):
        if int(seed) == seed:
            seed = int(seed)
        else:
            raise TypeError("Random seed must be an integer.")
    return seed

def check_unitary(U: np.ndarray, precision: float = 1e-10) -> bool:
    """
    A function to check if a provided matrix is unitary according to a 
    certain level of precision. If finds the product of the matrix with its
    hermitian conjugate and then checks it is unitary.
    
    Args:
    
        U (np.array) : The NxN matrix which we want to check is unitary.
        
        precision (float, optional) : The precision which the unitary
            matrix is checked according to. If there are large float errors 
            this may need to be reduced.
        
    Returns:
    
        bool : A boolean to indicate whether or not the matrix is unitary.
                            
    Raises:
    
        ValueError : Raised in the event that the matrix is not square as it
            cannot be unitary.
    
    """
    if U.shape[0] != U.shape[1]:
        raise ValueError("Unitary matrix must be square.")
    # Find hermitian conjugate and then product
    hc = np.conj(np.transpose(U))
    product = hc@U
    # To check if this it the identity then loop over each value and ensure
    # it is the expected number to some level of precision
    unitary = True
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            # Diagonal elements
            if i == j and (np.real(product[i,j] < 1-precision) or 
                np.imag(product[i,j]) > precision):
                unitary = False
            # Off diagonals
            elif i != j and (np.real(product[i,j] > precision) or 
                np.imag(product[i,j]) > precision):
                unitary = False
    # Return whether matrix is unitary or not
    return unitary

def add_mode_to_unitary(unitary: np.ndarray, add_mode: int) -> np.ndarray:
    """
    Adds a new mode (through inclusion of an extra row/column) to the provided 
    unitary at the selected location.
    
    Args:
    
        unitary (np.ndarray) : The unitary to add a mode to.
        
        add_mode (int) : The location at which a new mode should be added to 
            the circuit.
            
    Returns:
    
        np.ndarray : The converted unitary matrix.
    
    """
    dim = unitary.shape[0] + 1
    new_U = np.identity(dim, dtype = complex)
    # Diagonals
    new_U[:add_mode, :add_mode] = unitary[:add_mode, :add_mode]
    new_U[add_mode+1:, add_mode+1:] = unitary[add_mode:, add_mode:]
    # Off-diagonals
    new_U[:add_mode, add_mode+1:] = unitary[:add_mode, add_mode:]
    new_U[add_mode+1:, :add_mode] = unitary[add_mode:, :add_mode]
    return new_U