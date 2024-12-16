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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...__settings import settings
from ...sdk.circuit.photonic_compiler import CompiledPhotonicCircuit
from ...sdk.state import State
from .fock_backend import FockBackend


class SLOSBackend(FockBackend):
    """
    Contains calculate function for SLOS method.
    """

    @property
    def name(self) -> str:
        """Returns the name of the backend"""
        return "slos"

    def full_probability_distribution(
        self, circuit: CompiledPhotonicCircuit, input_state: State
    ) -> dict[State, float]:
        """
        Finds the output probability distribution for the provided circuit and
        input state.

        Args:

            circuit (CompiledPhotonicCircuit) : The compiled version of the
                circuit which is being simulated. This is created by calling the
                _build method on the target circuit.

            input_state (State) : The input state to the system.

        Returns:

            dict : The calculated full probability distribution for the input.

        Raises:

            BackendError: Raised if this method is called with an incompatible
                backend.

        """
        # Return empty distribution when 0 photons in input
        if input_state.n_photons == 0:
            return {State([0] * circuit.n_modes): 1.0}

        pdist: dict[State, float] = {}
        # Add extra states for loss modes here when included
        if circuit.loss_modes > 0:
            input_state = input_state + State([0] * circuit.loss_modes)
        full_dist = self.calculate(circuit.U_full, input_state)
        # Combine results to remote lossy modes
        for s, p in full_dist.items():
            if abs(p) ** 2 > settings.sampler_probability_threshold:
                new_s = State(s[: circuit.n_modes])  # type: ignore[arg-type]
                if new_s in pdist:
                    pdist[new_s] += abs(p) ** 2
                else:
                    pdist[new_s] = abs(p) ** 2
        return pdist

    def calculate(
        self, unitary: NDArray[np.complex128], input_state: State
    ) -> dict[tuple[int, ...], complex]:
        """
        Performs calculation of full probability distribution given a unitary
        matrix and input state.
        """
        p = [m for m, n in enumerate(input_state) for _i in range(n)]
        n_modes = unitary.shape[0]
        # Normalise initial input probability
        m = 1 / np.sqrt(vector_factorial(input_state.s))
        input_ = {tuple(n_modes * [0]): m}  # N-mode vacuum state

        # Successively apply the matrices A_k
        for i in p:  # Each matrix is indexed by the components of p
            output: dict[tuple[int, ...], complex] = {}
            for j in range(n_modes):  # Sum over i
                step = a_i_dagger(
                    input_, j, unitary[j, i]
                )  # Apply ladder operator
                output = add_dicts(output, step)  # Add it to the total
            input_ = output

        return input_


def a_i_dagger(
    dist: dict[tuple[int, ...], complex], mode: int, multiplier: complex
) -> dict[tuple[int, ...], complex]:
    """
    Ladder operator for the ith mode applied to the state v, where v is a
    dictionary
    """
    updated_dist = {}  # Create a new dictionary to store updated values

    for key, value in dist.items():
        key = list(key)  # type: ignore[assignment] # noqa: PLW2901
        key[mode] += 1  # type: ignore[index]
        # Update the new dictionary with modified key, value + normalisation
        updated_dist[tuple(key)] = key[mode] ** 0.5 * value * multiplier

    return updated_dist


def vector_factorial(vector: list[int]) -> int:
    """Calculates the product of factorials of the elements of the vector v"""
    return int(np.prod([factorial(i) for i in vector]))


def add_dicts(dict1: dict[Any, Any], dict2: dict[Any, Any]) -> dict[Any, Any]:
    """Function for combining two dictionaries together"""
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] += value
        else:
            dict1[key] = value
    return dict1
