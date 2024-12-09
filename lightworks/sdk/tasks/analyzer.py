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

from collections.abc import Callable

import numpy as np

from ...emulator.backends.fock_backend import FockBackend
from ..circuit import PhotonicCircuit
from ..results import SimulationResult
from ..state import State
from ..utils import (
    PhotonNumberError,
    add_heralds_to_state,
    fock_basis,
    process_post_selection,
)
from ..utils.post_selection import PostSelectionType
from .task import Task
from .task_utils import _check_photon_numbers, _validate_states


class Analyzer(Task):
    """
    The analyzer class is built as an alternative to simulation, intended for
    cases where we want to look at the transformations between a specific
    subset of states. It is useful for the simulation of probabilities in
    cases where loss and circuit errors are likely to be a factor. As part of
    the process a performance and error rate metric are calculated.

    Args:

        circuit (PhotonicCircuit) : The circuit to simulate.

        inputs (list) : A list of the input states to simulate. For multiple
            inputs this should be a list of States.

        expected (dict) : A dictionary containing a mapping between the input
            state and expected output state(s). If there is multiple
            possible outputs, this can be specified as a list.

    Attribute:

        performance : The total probabilities of mapping between the states
            provided compared with all possible states.

        error_rate : Given an expected mapping, the analyzer will determine the
            extent to which this is achieved.

    """

    __compatible_backends__ = ("permanent",)

    def __init__(
        self,
        circuit: PhotonicCircuit,
        inputs: State | list[State],
        expected: dict[State, State | list[State]] | None = None,
        post_selection: PostSelectionType | Callable | None = None,
    ) -> None:
        # Assign key parameters to attributes
        self.circuit = circuit
        self.post_selection = post_selection  # type: ignore[assignment]
        self.inputs = inputs  # type: ignore[assignment]
        self.expected = expected

        return

    @property
    def circuit(self) -> PhotonicCircuit:
        """
        Stores the circuit to be used for simulation, should be a
        PhotonicCircuit object.
        """
        return self.__circuit

    @circuit.setter
    def circuit(self, value: PhotonicCircuit) -> None:
        if not isinstance(value, PhotonicCircuit):
            raise TypeError(
                "Provided circuit should be a PhotonicCircuit or Unitary "
                "object."
            )
        self.__circuit = value

    @property
    def post_selection(self) -> PostSelectionType:
        """
        Stores post-selection criteria for analysis.
        """
        return self.__post_selection

    @post_selection.setter
    def post_selection(
        self, value: PostSelectionType | Callable | None
    ) -> None:
        value = process_post_selection(value)
        self.__post_selection = value

    @property
    def inputs(self) -> list[State]:
        """Store list of target inputs to the system."""
        return self.__inputs

    @inputs.setter
    def inputs(self, value: State | list[State]) -> None:
        self.__inputs = _validate_states(value, self.circuit.input_modes)

    @property
    def expected(self) -> dict[State, State | list[State]] | None:
        """
        A dictionary of the expected mapping between inputs and outputs of the
        system.
        """
        return self.__expected

    @expected.setter
    def expected(self, value: dict[State, State | list[State]] | None) -> None:
        self.__expected = value

    def _run(self, backend: FockBackend) -> SimulationResult:  # type: ignore[override]
        """
        Function to perform analysis of probabilities between different
        inputs/outputs.

        Args:

            backend (FockBackend) : The target backend to run the task with.

        Returns:

            SimulationResult : A dictionary containing an array of probability
                values between the provided inputs/outputs.

        """
        self.__backend = backend
        self.__circuit_built = self.circuit._build()
        n_modes = self.circuit.input_modes
        if self.circuit.heralds["input"] != self.circuit.heralds["output"]:
            raise RuntimeError(
                "Mismatch in number of heralds on the input/output modes, it "
                "is likely this results from a herald being added twice or "
                "modified."
            )
        # Process inputs, adding heralds and loss modes
        inputs = list(self.inputs)  # Copy input list
        _check_photon_numbers(inputs)
        in_heralds = self.circuit.heralds["input"]
        full_inputs = [
            add_heralds_to_state(i, in_heralds)
            + [0] * self.__circuit_built.loss_modes
            for i in inputs
        ]
        n_photons = sum(full_inputs[0])
        # Generate lists of possible outputs with and without heralded modes
        full_outputs, filtered_outputs = self._generate_outputs(
            n_modes, n_photons
        )
        # Calculate permanent for the given inputs and outputs and return
        # values
        probs = self._get_probs(full_inputs, full_outputs)
        # Calculate performance by finding sum of valid transformations
        self.performance = probs.sum() / len(full_inputs)
        # Analyse error rate from expected results if specified
        if self.expected is not None:
            self.error_rate = self._calculate_error_rate(
                probs, inputs, filtered_outputs, self.expected
            )
        # Compile results into results object
        results = SimulationResult(
            probs,
            "probability",
            inputs=inputs,
            outputs=filtered_outputs,
            performance=self.performance,
        )
        if hasattr(self, "error_rate"):
            results.error_rate = self.error_rate  # type: ignore
        self.results = results
        # Return dict
        return results

    def _get_probs(
        self, full_inputs: list[list[int]], full_outputs: list[list[int]]
    ) -> np.ndarray:
        """
        Create an array of output probabilities for a given set of inputs and
        outputs.
        """
        probs = np.zeros((len(full_inputs), len(full_outputs)))
        for i, ins in enumerate(full_inputs):
            for j, outs in enumerate(full_outputs):
                # No loss case
                if not self.__circuit_built.loss_modes:
                    probs[i, j] += self.__backend.probability(
                        self.__circuit_built.U_full, ins, outs
                    )
                # Lossy case
                else:
                    # Photon number preserved
                    if sum(ins) == sum(outs):
                        outs = (  # noqa: PLW2901
                            outs + [0] * self.__circuit_built.loss_modes
                        )
                        probs[i, j] += self.__backend.probability(
                            self.__circuit_built.U_full, ins, outs
                        )
                    # Otherwise
                    else:
                        # If n_out < n_in work out all loss mode combinations
                        # and find probability of each
                        n_loss = sum(ins) - sum(outs)
                        if n_loss < 0:
                            raise PhotonNumberError(
                                "Output photon number larger than input "
                                "number."
                            )
                        # Find loss states and find probability of each
                        loss_states = fock_basis(
                            self.__circuit_built.loss_modes, n_loss
                        )
                        for ls in loss_states:
                            fs = outs + ls
                            probs[i, j] += self.__backend.probability(
                                self.__circuit_built.U_full, ins, fs
                            )

        return probs

    def _calculate_error_rate(
        self,
        probabilities: np.ndarray,
        inputs: list[State],
        outputs: list[State],
        expected: dict[State, State | list[State]],
    ) -> float:
        """
        Calculate the error rate for a set of expected mappings between inputs
        and outputs, given the results calculated by the analyzer.
        """
        # Check all inputs in expectation mapping
        for s in inputs:
            if s not in expected:
                msg = f"Input state {s} not in provided expectation dict."
                raise KeyError(msg)
        # For each input check error rate
        errors = []
        for i, s in enumerate(inputs):
            out = expected[s]
            # Convert expected output to list if only one value provided
            if isinstance(out, State):
                out = [out]
            iprobs = probabilities[i, :]
            error = 1
            # Loop over expected outputs and subtract from error value
            for o in out:
                if o in outputs:
                    loc = outputs.index(o)
                    error -= iprobs[loc] / sum(iprobs)
            errors += [error]
        # Then take average and return
        return float(np.mean(errors))

    def _generate_outputs(
        self, n_modes: int, n_photons: int
    ) -> tuple[list[list[int]], list[State]]:
        """
        Generates all possible outputs for a given number of modes, photons and
        heralding + post-selection conditions. It returns two list, one with
        the heralded modes included and one without.
        """
        # Get all possible outputs for the non-herald modes
        if not self.__circuit_built.loss_modes:
            outputs = fock_basis(n_modes, n_photons)
        # Combine all n < n_in for lossy case
        else:
            outputs = []
            for n in range(n_photons + 1):
                outputs += fock_basis(n_modes, n)
        # Filter outputs according to post selection and add heralded photons
        filtered_outputs = []
        full_outputs = []
        out_heralds = self.circuit.heralds["output"]
        for state in outputs:
            # Check output meets all post selection rules
            if self.post_selection.validate(state):
                filtered_outputs += [State(state)]
                full_outputs += [add_heralds_to_state(state, out_heralds)]
        # Check some valid outputs found
        if not full_outputs:
            raise ValueError(
                "No valid outputs found, consider relaxing post-selection."
            )

        return (full_outputs, filtered_outputs)
