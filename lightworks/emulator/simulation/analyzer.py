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

from typing import Callable

import numpy as np

from ...sdk.circuit import Circuit
from ...sdk.state import State
from ...sdk.utils import add_heralds_to_state
from ...sdk.utils.post_selection import PostSelectionType
from ..backend import Backend
from ..results import SimulationResult
from ..utils import (
    ModeMismatchError,
    PhotonNumberError,
    fock_basis,
    process_post_selection,
)


class Analyzer:
    """
    The analyzer class is built as an alternative to simulation, intended for
    cases where we want to look at the transformations between a specific
    subset of states. It is useful for the simulation of probabilities in
    cases where loss and circuit errors are likely to be a factor. As part of
    the process a performance and error rate metric are calculated.

    Args:

        circuit (Circuit) : The circuit to simulate.

    Attribute:

        performance : The total probabilities of mapping between the states
            provided compared with all possible states.

        error_rate : Given an expected mapping, the analyzer will determine the
            extent to which this is achieved.

    """

    def __init__(self, circuit: Circuit) -> None:
        # Assign key parameters to attributes
        self.circuit = circuit
        self.post_selection = None  # type: ignore
        self.__backend = Backend("permanent")

        return

    @property
    def circuit(self) -> Circuit:
        """
        Stores the circuit to be used for simulation, should be a Circuit
        object.
        """
        return self.__circuit

    @circuit.setter
    def circuit(self, value: Circuit) -> None:
        if not isinstance(value, Circuit):
            raise TypeError(
                "Provided circuit should be a Circuit or Unitary object."
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

    def analyze(
        self, inputs: State | list, expected: dict | None = None
    ) -> SimulationResult:
        """
        Function to perform analysis of probabilities between
        different inputs/outputs

        Args:

            inputs (list) : A list of the input states to simulate. For
                multiple inputs this should be a list of States.

            expected (dict) : A dictionary containing a mapping between the
                input state and expected output state(s). If there is multiple
                possible outputs, this can be specified as a list.

        Returns:

            SimulationResult : A dictionary containing an array of probability
                values between the provided inputs/outputs.

        """
        self.__circuit_built = self.circuit._build()
        n_modes = self.circuit.input_modes
        if self.circuit.heralds["input"] != self.circuit.heralds["output"]:
            raise RuntimeError(
                "Mismatch in number of heralds on the input/output modes, it "
                "is likely this results from a herald being added twice or "
                "modified."
            )
        # Convert state to list of States if not provided for single state case
        if isinstance(inputs, State):
            inputs = [inputs]
        # Process inputs using dedicated function
        full_inputs = self._process_inputs(inputs)
        n_photons = full_inputs[0].n_photons
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
        if expected is not None:
            self.error_rate = self._calculate_error_rate(
                probs, inputs, filtered_outputs, expected
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

    def _get_probs(self, full_inputs: list, full_outputs: list) -> np.ndarray:
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
                        self.__circuit_built.U_full, ins.s, outs
                    )
                # Lossy case
                else:
                    # Photon number preserved
                    if ins.n_photons == sum(outs):
                        outs = (  # noqa: PLW2901
                            outs + [0] * self.__circuit_built.loss_modes
                        )
                        probs[i, j] += self.__backend.probability(
                            self.__circuit_built.U_full, ins.s, outs
                        )
                    # Otherwise
                    else:
                        # If n_out < n_in work out all loss mode combinations
                        # and find probability of each
                        n_loss = ins.n_photons - sum(outs)
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
                                self.__circuit_built.U_full, ins.s, fs
                            )

        return probs

    def _calculate_error_rate(
        self,
        probabilities: np.ndarray,
        inputs: list,
        outputs: list,
        expected: dict,
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

    def _process_inputs(self, inputs: list) -> list[State]:
        """
        Takes the provided inputs, perform error checking on them and adds any
        heralded photons that are required, returning full states..
        """
        n_modes = self.circuit.input_modes
        # Ensure all photon numbers are the same
        ns = [s.n_photons for s in inputs]
        if min(ns) != max(ns):
            raise PhotonNumberError("Mismatch in photon numbers between inputs")
        full_inputs = []
        in_heralds = self.circuit.heralds["input"]
        # Check dimensions of input and add heralded photons
        for state in inputs:
            if len(state) != n_modes:
                raise ModeMismatchError(
                    "Input states are of the wrong dimension. Remember to "
                    "subtract heralded modes."
                )
            # Also validate state values
            state._validate()
            full_inputs += [State(add_heralds_to_state(state, in_heralds))]
        # Add extra states for loss modes here when included
        if self.__circuit_built.loss_modes > 0:
            full_inputs = [
                s + State([0] * self.__circuit_built.loss_modes)
                for s in full_inputs
            ]
        return full_inputs

    def _generate_outputs(
        self, n_modes: int, n_photons: int
    ) -> tuple[list, list]:
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
                fo = add_heralds_to_state(state, out_heralds)
                filtered_outputs += [State(state)]
                full_outputs += [fo]
        # Check some valid outputs found
        if not full_outputs:
            raise ValueError(
                "No valid outputs found, consider relaxing post-selection."
            )

        return (full_outputs, filtered_outputs)
