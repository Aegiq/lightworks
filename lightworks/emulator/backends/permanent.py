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
from thewalrus import perm

from ...__settings import settings
from ...sdk.circuit.compiler import CompiledCircuit
from ...sdk.state import State
from ..utils import fock_basis
from .abc_backend import BackendABC


class PermanentBackend(BackendABC):
    """
    Calculate the permanent for a give unitary matrix and input state. In this
    case, thewalrus module is used for all permanent calculations.
    """

    @property
    def name(self) -> str:
        """Returns the name of the backend"""
        return "permanent"

    def probability_amplitude(
        self, unitary: np.ndarray, input_state: list, output_state: list
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

        Raises:

            BackendError: Raised if this method is called with an incompatible
                backend.

        """
        factor_m = prod([factorial(i) for i in input_state])
        factor_n = prod([factorial(i) for i in output_state])
        # Calculate permanent for given input/output
        return perm(partition(unitary, input_state, output_state)) / (
            np.sqrt(factor_m * factor_n)
        )

    def probability(
        self, unitary: np.ndarray, input_state: list, output_state: list
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

        Raises:

            BackendError: Raised if this method is called with an incompatible
                backend.

        """
        return (
            abs(self.probability_amplitude(unitary, input_state, output_state))
            ** 2
        )

    def full_probability_distribution(
        self, circuit: CompiledCircuit, input_state: State
    ) -> dict:
        """
        Finds the output probability distribution for the provided circuit and
        input state.

        Args:

            circuit (CompiledCircuit) : The compiled version of the circuit
                which is being simulated. This is created by calling the _build
                method on the target circuit.

            input_state (State) : The input state to the system.

        Returns:

            dict : The calculated full probability distribution for the input.

        Raises:

            BackendError: Raised if this method is called with an incompatible
                backend.

        """
        pdist: dict[State, float] = {}
        # Return empty distribution when 0 photons in input
        if input_state.n_photons == 0:
            pdist = {State([0] * circuit.n_modes): 1.0}

        # Add extra states for loss modes here when included
        if circuit.loss_modes > 0:
            input_state = input_state + State([0] * circuit.loss_modes)
        # For a given input work out all possible outputs
        out_states = fock_basis(len(input_state), input_state.n_photons)
        for ostate in out_states:
            # Skip any zero photon states
            if sum(ostate[: circuit.n_modes]) == 0:
                continue
            p = self.probability(circuit.U_full, input_state.s, ostate)
            if p > settings.sampler_probability_threshold:
                # Only care about non-loss modes
                ostate = State(ostate[: circuit.n_modes])  # noqa: PLW2901
                if ostate in pdist:
                    pdist[ostate] += p
                else:
                    pdist[ostate] = p
        # Work out zero photon component before saving to unique results
        total_prob = sum(pdist.values())
        if total_prob < 1 and circuit.loss_modes > 0:
            pdist[State([0] * circuit.n_modes)] = 1 - total_prob

        return pdist


def partition(
    unitary: np.ndarray, in_state: list, out_state: list
) -> np.ndarray:
    """
    Converts the unitary matrix into a larger matrix used for in the
    permanent calculation.
    """
    n_modes = len(in_state)  # Number of modes
    # Construct the matrix of indices for the partition
    x, y = [], []
    for i in range(n_modes):
        x += [i] * out_state[i]
        y += [i] * in_state[i]
    # Construct the new matrix with dimension n, where n is photon number
    return unitary[np.ix_(x, y)]
