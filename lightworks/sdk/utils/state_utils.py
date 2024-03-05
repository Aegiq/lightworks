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
Script to store various useful functions for the simulation aspect of the code.
"""

def fock_basis(N: int, n: int, statistic_type: str) -> list:
    """
    Returns the Fock basis for n photons in N modes for either bosonic of
    fermionic statistics.
    """
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
                
def state_to_string(state: list) -> str:
    """Converts the provided state to a string with ket notation."""
    string = "|"
    for s in state:
        string += str(s) + ","
    string = string[:-1] + ">"
    return string