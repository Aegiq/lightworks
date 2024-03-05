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

from ...sdk import State

import numpy as np

class Determinant:
    """
    Determinant class
    This class contains the functions required to calculate the probability 
    amplitude of transition between a given input and out state for the 
    specified unitary matrix in cases of fermionic statistics. 
    """
    
    @staticmethod
    def calculate(U: np.ndarray, in_state: State, out_state: State) -> complex: 
        """
        Function to calculate the determinant for a given unitary, input state
        and output state. It returns the complex probability amplitude for the
        state.
        """
        # Calculate determinant for given input/output
        return np.linalg.det(Determinant.partition(U, in_state, out_state))

    @staticmethod
    def partition(U: np.ndarray, in_state: State, 
                  out_state: State) -> np.ndarray:
        """
        Converts the unitary matrix into a larger matrix used for in the 
        determinant calculation.
        """
        N = len(in_state) # Number of modes
        # Construct the matrix of indices for the partition
        X, Y = [], []
        for i in range(N):
            X += [i]*out_state[i]
            Y += [i]*in_state[i]
        # Construct the new matrix with dimension n, where n is photon number 
        part_U = U[np.ix_(X, Y)] 
        
        return part_U