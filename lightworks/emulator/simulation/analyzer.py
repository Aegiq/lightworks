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
from numpy.typing import NDArray

from ...sdk.results import SimulationResult
from ...sdk.state import State
from ...sdk.tasks import AnalyzerTask
from ...sdk.utils import (
    DefaultPostSelection,
    PhotonNumberError,
    add_heralds_to_state,
)
from ..utils import check_photon_numbers, fock_basis
from .runner import RunnerABC


class AnalyzerRunner(RunnerABC):
    """
    Calculates an output probability distribution under the assigned set of
    post-selection + heralding constraints.

    Args:

        data (AnalyzerTask) : The task which is to be executed.

        probability_function (Callable) : Function for calculating probability
            of transition between an input and output for a given unitary.

    """

    def __init__(
        self,
        data: AnalyzerTask,
        probability_function: Callable[
            [NDArray[np.complex128], list[int], list[int]], float
        ],
    ) -> None:
        self.data = data
        self.func = probability_function

    def run(self) -> SimulationResult:
        """
        Runs analyzer process using the configured data.

        Returns:

            SimulationResult : A dictionary containing an array of probability
                values between the provided inputs/outputs.

        """
        n_modes = self.data.circuit.input_modes
        if (
            self.data.circuit.heralds["input"]
            != self.data.circuit.heralds["output"]
        ):
            raise RuntimeError(
                "Mismatch in number of heralds on the input/output modes, it "
                "is likely this results from a herald being added twice or "
                "modified."
            )
        # Process inputs, adding heralds and loss modes
        inputs = list(self.data.inputs)  # Copy input list
        check_photon_numbers(inputs)
        in_heralds = self.data.circuit.heralds["input"]
        full_inputs = [
            add_heralds_to_state(i, in_heralds)
            + [0] * self.data.circuit.loss_modes
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
        performance = probs.sum() / len(full_inputs)
        # Analyse error rate from expected results if specified
        if self.data.expected is not None:
            error_rate = self._calculate_error_rate(
                probs, inputs, filtered_outputs, self.data.expected
            )
        else:
            error_rate = None
        # Compile results into results object
        results = SimulationResult(
            probs,
            "probability",
            inputs=inputs,
            outputs=filtered_outputs,
            performance=performance,
        )
        if error_rate is not None:
            results.error_rate = error_rate  # type: ignore
        # Return dict
        return results

    def _get_probs(
        self, full_inputs: list[list[int]], full_outputs: list[list[int]]
    ) -> NDArray[np.float64]:
        """
        Create an array of output probabilities for a given set of inputs and
        outputs.
        """
        probs = np.zeros((len(full_inputs), len(full_outputs)))
        for i, ins in enumerate(full_inputs):
            for j, outs in enumerate(full_outputs):
                # No loss case
                if not self.data.circuit.loss_modes:
                    probs[i, j] += self.func(
                        self.data.circuit.U_full, ins, outs
                    )
                # Lossy case
                else:
                    # Photon number preserved
                    if sum(ins) == sum(outs):
                        outs = (  # noqa: PLW2901
                            outs + [0] * self.data.circuit.loss_modes
                        )
                        probs[i, j] += self.func(
                            self.data.circuit.U_full, ins, outs
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
                            self.data.circuit.loss_modes, n_loss
                        )
                        for ls in loss_states:
                            fs = outs + ls
                            probs[i, j] += self.func(
                                self.data.circuit.U_full, ins, fs
                            )

        return probs

    def _calculate_error_rate(
        self,
        probabilities: NDArray[np.float64],
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
        if not self.data.circuit.loss_modes:
            outputs = fock_basis(n_modes, n_photons)
        # Combine all n < n_in for lossy case
        else:
            outputs = []
            for n in range(n_photons + 1):
                outputs += fock_basis(n_modes, n)
        # Filter outputs according to post selection and add heralded photons
        filtered_outputs = []
        full_outputs = []
        out_heralds = self.data.circuit.heralds["output"]
        post_selection = (
            DefaultPostSelection()
            if self.data.post_selection is None
            else self.data.post_selection
        )
        for state in outputs:
            # Check output meets all post selection rules
            if post_selection.validate(state):
                filtered_outputs += [State(state)]
                full_outputs += [add_heralds_to_state(state, out_heralds)]
        # Check some valid outputs found
        if not full_outputs:
            raise ValueError(
                "No valid outputs found, consider relaxing post-selection."
            )

        return (full_outputs, filtered_outputs)
