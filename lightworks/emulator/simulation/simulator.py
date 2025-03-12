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

from lightworks.emulator.utils.sim import check_photon_numbers
from lightworks.emulator.utils.state import fock_basis
from lightworks.sdk.results import SimulationResult
from lightworks.sdk.state import State
from lightworks.sdk.tasks import SimulatorTask
from lightworks.sdk.utils.heralding import add_heralds_to_state

from .runner import RunnerABC


class SimulatorRunner(RunnerABC):
    """
    Calculates the probability amplitudes between a set of inputs and outputs
    from a given circuit.

    Args:

        data (SimulatorTask) : The task which is to be executed.

        amplitude_function (Callable) : Function for calculating probability
            amplitudes between an input and output for a given unitary.

    """

    def __init__(
        self,
        data: SimulatorTask,
        amplitude_function: Callable[
            [NDArray[np.complex128], list[int], list[int]], complex
        ],
    ) -> None:
        self.data = data
        self.func = amplitude_function

    def run(self) -> SimulationResult:
        """
        Runs the simulation task.

        Returns:

            SimulationResult : A dictionary containing the calculated
                probability amplitudes, where the first index of the array
                corresponds to the input state, as well as the input and output
                state used to create the array.

        """
        in_heralds = self.data.circuit.heralds.input
        out_heralds = self.data.circuit.heralds.output
        in_heralds_n = sum(in_heralds.values())
        out_heralds_n = sum(out_heralds.values())
        target_n = self.data.inputs[0].n_photons + in_heralds_n
        if self.data.outputs is None:
            check_photon_numbers(self.data.inputs, target_n - in_heralds_n)
            outputs = [
                State(s)
                for s in fock_basis(
                    self.data.circuit.input_modes, target_n - out_heralds_n
                )
            ]
        else:
            check_photon_numbers(self.data.inputs, target_n - in_heralds_n)
            check_photon_numbers(self.data.outputs, target_n - out_heralds_n)
            outputs = self.data.outputs
        # Pre-add output values to avoid doing this many times
        full_outputs = [
            add_heralds_to_state(outs, out_heralds)
            + [0] * self.data.circuit.loss_modes
            for outs in outputs
        ]
        # Calculate permanent for the given inputs and outputs and return
        # values
        amplitudes = np.zeros(
            (len(self.data.inputs), len(outputs)), dtype=complex
        )
        for i, ins in enumerate(self.data.inputs):
            in_state = add_heralds_to_state(ins, in_heralds)
            in_state += [0] * self.data.circuit.loss_modes
            for j, outs in enumerate(full_outputs):
                amplitudes[i, j] = self.func(
                    self.data.circuit.U_full, in_state, outs
                )
        # Return results and corresponding states as dictionary
        return SimulationResult(
            amplitudes,
            "probability_amplitude",
            inputs=self.data.inputs,
            outputs=outputs,
        )
