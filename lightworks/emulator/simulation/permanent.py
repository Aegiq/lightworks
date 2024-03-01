from ...sdk import State

import numpy as np
from thewalrus import perm
from math import factorial, prod

class Permanent:
    """
    Permanent class
    This class contains the functions required to calculate the permanent of a
    matrix. In this case, thewalrus module is used for all permanent 
    calculations.
    """
    
    @staticmethod
    def calculate(U: np.ndarray, in_state: State, out_state: State) -> complex: 
        """
        Function to calculate the permanent for a given unitary, input state
        and output state. It returns the complex probability amplitude for the
        state.
        """
        factor_m = prod([factorial(i) for i in in_state])
        factor_n = prod([factorial(i) for i in out_state])
        # Calculate permanent for given input/output
        p = (perm(Permanent.partition(U, in_state, out_state)) / 
             (np.sqrt(factor_m*factor_n)))    
        return p

    @staticmethod
    def partition(U: np.ndarray, in_state: State, 
                  out_state: State) -> np.ndarray:
        """
        Converts the unitary matrix into a larger matrix used for in the 
        permanent calculation.
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