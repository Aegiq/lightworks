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

from math import factorial

import numpy as np

from ...sdk.state import State


class SLOS:
    """
    Contains calculate function for SLOS method.
    """

    @staticmethod
    def calculate(unitary: np.ndarray, input_state: State) -> dict:
        """
        Performs calculation of full probability distribution given a unitary
        matrix and input state.
        """
        p = [m for m, n in enumerate(input_state) for _i in range(n)]
        n_modes = unitary.shape[0]
        input = {tuple(n_modes * [0]): 1.0}  # N-mode vacuum state

        # Successively apply the matrices A_k
        for i in p:  # Each matrix is indexed by the components of p
            output: dict[tuple, float] = {}
            for j in range(n_modes):  # Sum over i
                step = a_i_dagger(
                    input, j, unitary[j, i]
                )  # Apply ladder operator
                output = add_dicts(output, step)  # Add it to the total
            input = output

        # Renormalise the output with the overall factorial term and return
        m = 1 / np.sqrt(vector_factorial(input_state.s))
        return {k: v * m for k, v in input.items()}


def a_i_dagger(dist: dict, mode: int, multiplier: complex) -> dict:
    """
    Ladder operator for the ith mode applied to the state v, where v is a
    dictionary
    """
    updated_dist = {}  # Create a new dictionary to store updated values

    for key, value in dist.items():
        key = list(key)  # noqa: PLW2901
        key[mode] += 1  # Increase the number of photons in the ith mode by 1
        # Update the new dictionary with modified key, value + normalisation
        updated_dist[tuple(key)] = key[mode] ** 0.5 * value * multiplier

    return updated_dist


def vector_factorial(vector: list) -> int:
    """Calculates the product of factorials of the elements of the vector v"""
    return int(np.prod([factorial(i) for i in vector]))


def add_dicts(dict1: dict, dict2: dict) -> dict:
    """Function for combining two dictionaries together"""
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] += value
        else:
            dict1[key] = value
    return dict1
