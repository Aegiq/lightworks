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
        
        """
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