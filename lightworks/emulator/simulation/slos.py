from ...sdk import State

import numpy as np
import math

class SLOS:
    """
    Contains calculate function for SLOS method.
    """
    @staticmethod
    def calculate(U, input_state: State) -> dict:

        p = []
        for m, n in enumerate(input_state):
            for i in range(n):
                p.append(m)
        N = U.shape[0]
        input = {tuple(N*[0]):1} #N-mode vacuum state

        # Successively apply the matrices A_k
        for k in p: # Each matrix is indexed by the components of p
            output = {}
            for i in range(N):  # Sum over i
                step_1 = a_i_dagger(i, input) # Apply ladder operator
                step_2 = dict_scale(step_1, U[i,k]) # Multiply by the unitary matrix component
                output = add_dicts(output, step_2) # Add it to the total
            input = output

        # Renormalise the output with the overall factorial term and return
        return dict_scale(input,1/np.sqrt(vector_factorial(input_state)))

def a_i_dagger(i,v):
    """
    Ladder operator for the ith mode applied to the state v, where v is a 
    dictionary
    """
    
    updated_v = {}  # Create a new dictionary to store updated keys and values

    for key, value in v.items():
        key = list(key)
        key[i] += 1 # Increase the number of photons in the ith mode by 1
        updated_v[tuple(key)] = np.sqrt(key[i])*value  # Update the new dictionary with modified key and value, with the normalisation factor

    return updated_v

def dict_scale(x,a):
    """Scales all entries of a dictionary by the factor a"""
    y = {k:a*v for k, v in x.items()}
    return y

def vector_factorial(v):
    """Calculates the product of factorials of the elements of the vector v"""
    c = 1
    for i in v:
        c = c*math.factorial(i)
    return c

def add_dicts(dict1, dict2):
    """Adds two dictionaries together"""
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] += value  # If key exists in dict1, add the value to it
        else:
            dict1[key] = value  # If key does not exist in dict1, add it with its value
    return dict1