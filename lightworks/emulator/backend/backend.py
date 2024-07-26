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

from numpy import ndarray

from ...sdk.circuit.compiler import CompiledCircuit
from ...sdk.state import State
from ..utils import BackendError, fock_basis
from .permanent import Permanent
from .slos import SLOS


class Backend:
    """
    Provide central location for selecting and interacting with different
    simulation backends.

    Args:

        backend (str) : A string detailing the backend which is to be used.

    """

    def __init__(self, backend: str) -> None:
        self.backend = backend

        return

    @property
    def backend(self) -> str:
        """Stores data on the selected backend."""
        return self.__backend

    @backend.setter
    def backend(self, value: str) -> None:
        if value not in ["permanent", "slos", "clifford"]:
            raise ValueError("Invalid backend provided.")
        # Temporary extra checking
        if value in ["clifford"]:
            raise NotImplementedError(
                "Support for this backend has not yet been included."
            )
        self.__backend = value

    def __str__(self) -> str:
        return self.backend

    def __repr__(self) -> str:
        return f"lightworks.emulator.Backend('{self.backend}')"

    def probability_amplitude(
        self, unitary: ndarray, input_state: list, output_state: list
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
        if self.backend != "permanent":
            raise BackendError(
                "Direct probability amplitude calculation only supported for "
                "permanent backend."
            )
        return Permanent.calculate(unitary, input_state, output_state)

    def probability(
        self, unitary: ndarray, input_state: list, output_state: list
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
        # Otherwise vary distribution calculation method
        elif self.backend == "permanent":
            # Add extra states for loss modes here when included
            if circuit.loss_modes > 0:
                input_state = input_state + State([0] * circuit.loss_modes)
            # For a given input work out all possible outputs
            out_states = fock_basis(len(input_state), input_state.n_photons)
            for ostate in out_states:
                # Skip any zero photon states
                if sum(ostate[: circuit.n_modes]) == 0:
                    continue
                p = Permanent.calculate(circuit.U_full, input_state.s, ostate)
                if abs(p) ** 2 > 0:
                    # Only care about non-loss modes
                    ostate = State(ostate[: circuit.n_modes])  # noqa: PLW2901
                    if ostate in pdist:
                        pdist[ostate] += abs(p) ** 2
                    else:
                        pdist[ostate] = abs(p) ** 2
            # Work out zero photon component before saving to unique results
            total_prob = sum(pdist.values())
            if total_prob < 1 and circuit.loss_modes > 0:
                pdist[State([0] * circuit.n_modes)] = 1 - total_prob
        elif self.backend == "slos":
            # Add extra states for loss modes here when included
            if circuit.loss_modes > 0:
                input_state = input_state + State([0] * circuit.loss_modes)
            full_dist = SLOS.calculate(circuit.U_full, input_state)
            # Combine results to remote lossy modes
            for s, p in full_dist.items():
                new_s = State(s[: circuit.n_modes])
                if new_s in pdist:
                    pdist[new_s] += abs(p) ** 2
                else:
                    pdist[new_s] = abs(p) ** 2
        elif self.backend == "clifford":
            raise BackendError(
                "Probability distribution calculation not supported for "
                "clifford backend."
            )

        return pdist
