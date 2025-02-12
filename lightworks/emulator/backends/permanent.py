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


from math import factorial, prod

import numpy as np
from numpy.typing import NDArray
from thewalrus import perm

from lightworks.__settings import settings
from lightworks.emulator.utils.state import fock_basis
from lightworks.sdk.circuit.photonic_compiler import CompiledPhotonicCircuit
from lightworks.sdk.state import State

from .fock_backend import FockBackend


class PermanentBackend(FockBackend):
    """
    Calculate the permanent for a give unitary matrix and input state. In this
    case, thewalrus module is used for all permanent calculations.
    """

    @property
    def name(self) -> str:
        """Returns the name of the backend"""
        return "permanent"

    @property
    def compatible_tasks(self) -> tuple[str, ...]:
        """Returns backends which are compatible with the backend."""
        return ("Sampler", "Analyzer", "Simulator")

    def probability_amplitude(
        self,
        unitary: NDArray[np.complex128],
        input_state: list[int],
        output_state: list[int],
    ) -> complex:
        """
        Find the probability amplitude between a given input and output state
        for the provided unitary. Note values should be provided as an
        array/list.

        Args:

            unitary (np.ndarray) : The target unitary matrix which represents
                the transformation implemented by a circuit.

            input_state (list) : The input state to the system.

            output_state (list) : The target output state.

        Returns:

            complex : The calculated probability amplitude.

        """
        factor_m = prod([factorial(i) for i in input_state])
        factor_n = prod([factorial(i) for i in output_state])
        # Calculate permanent for given input/output
        return perm(partition(unitary, input_state, output_state)) / (
            np.sqrt(factor_m * factor_n)
        )

    def probability(
        self,
        unitary: NDArray[np.complex128],
        input_state: list[int],
        output_state: list[int],
    ) -> float:
        """
        Calculates the probability of a given output state for a provided
        unitary and input state to the system. Note values should be provided
        as an array/list.

        Args:

            unitary (np.ndarray) : The target unitary matrix which represents
                the transformation implemented by a circuit.

            input_state (list) : The input state to the system.

            output_state (list) : The target output state.

        Returns:

            float : The calculated probability of transition between the input
                and output.

        """
        return (
            abs(self.probability_amplitude(unitary, input_state, output_state))
            ** 2
        )

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

        """
        # Return empty distribution when 0 photons in input
        if input_state.n_photons == 0:
            return {State([0] * circuit.n_modes): 1.0}

        pdist: dict[State, float] = {}
        # Add extra states for loss modes here when included
        if circuit.loss_modes > 0:
            input_state += State([0] * circuit.loss_modes)
        # For a given input work out all possible outputs
        out_states = fock_basis(len(input_state), input_state.n_photons)
        for ostate in out_states:
            # Skip any zero photon states
            if sum(ostate[: circuit.n_modes]) == 0:
                continue
            p = self.probability(circuit.U_full, input_state.s, ostate)
            if p > settings.sampler_probability_threshold:
                # Only care about non-loss modes
                meas_state = State(ostate[: circuit.n_modes])
                if meas_state in pdist:
                    pdist[meas_state] += p
                else:
                    pdist[meas_state] = p
        # Work out zero photon component before saving to unique results
        total_prob = sum(pdist.values())
        if total_prob < 1 and circuit.loss_modes > 0:
            pdist[State([0] * circuit.n_modes)] = 1 - total_prob

        return pdist


def partition(
    unitary: NDArray[np.complex128], in_state: list[int], out_state: list[int]
) -> NDArray[np.complex128]:
    """
    Converts the unitary matrix into a larger matrix used for in the
    permanent calculation.
    """
    n_modes = len(in_state)  # Number of modes
    # Construct the matrix of indices for the partition
    x = [i for i in range(n_modes) for _ in range(out_state[i])]
    y = [i for i in range(n_modes) for _ in range(in_state[i])]
    # Construct the new matrix with dimension n, where n is photon number
    return unitary[np.ix_(x, y)]
