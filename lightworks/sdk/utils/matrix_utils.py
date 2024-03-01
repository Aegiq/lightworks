"""
This file contains a collection of different useful functions for operations on
matrices.
"""

import numpy as np
from scipy.stats import unitary_group
from typing import Any

def fidelity(U_target: np.ndarray, U_calculated: np.ndarray) -> float:
    """
    Function to calculate the fidelity between target and calculated unitary
    matrices. Note: It is important that these two matrices both either contain
    probability amplitudes or normalised probabilities.
    
    Args:
    
        U_target (np.ndarray) : The target unitary matrix.
        
        U_calculated (np.ndarray) : The calculated unitary matrix from the 
            simulation.
    
    Returns:
    
        float : The calculated fidelity between the two matrices.
        
    """
    U_target, U_calculated = np.array(U_target), np.array(U_calculated)
    N = U_target.shape[0]
    # Find h.c. of calculated unitary to condense code
    Udag = np.conj(np.transpose(U_calculated))
    # Calculate fidelity using the unitary matrices
    fidelity = (abs(np.trace(Udag @ U_target)) ** 2 / 
                (N * np.trace(Udag @ U_calculated)))
    return float(np.real(fidelity))

def random_unitary(N: int, seed: int = None) -> np.ndarray:
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

def random_permutation(N: int, seed: int = None) -> np.ndarray:
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