"""
Script to store various useful functions for the simulation aspect of the code.
"""

from numpy import zeros, ones, concatenate

def fock_basis(N: int, n: int, statistic_type: str) -> list:
    """Returns the Fock basis for n photons in N modes for either bosonic of
    fermionic statistics."""
    if statistic_type == "bosonic":
        return list(_sums(N,n))
    elif statistic_type == "fermionic":
        return _fermionic_basis(N, n)
    else:
        raise ValueError("statistic_type should be 'bosonic' or 'fermionic'.")
    
def _fermionic_basis(N,n):
    """This returns the possible states of n fermions in N modes as vectors."""
    if n == 0:
        return [[0]*N]
    if N == n:
        return [[1]*N]
    arrays_with_zero = [[0]+arr for arr in _fermionic_basis(N - 1, n)]
    arrays_with_one = [[1]+arr for arr in _fermionic_basis(N - 1, n - 1)]
    return arrays_with_zero + arrays_with_one

def _sums(length, total_sum):
    if length == 1:
        yield [total_sum,]
    else:
        for value in range(total_sum + 1):
            for permutation in _sums(length-1, total_sum-value):
                yield permutation + [value,]
                
def annotated_state_to_string(state: list) -> str:
    """Converts the provided annotated state to a string with ket notation."""
    string = "|"
    for s in state:
        if len(s) > 0:
            string += str(len(s)) + ":("
            for label in s:
                string += str(label) + ","
            string = string[:-1] + "),"
        else:
            string += "0,"
    string = string[:-1] + ">"
    return string