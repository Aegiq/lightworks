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


from dataclasses import dataclass

import numpy as np

from ...emulator.backends.fock_backend import FockBackend
from ..circuit import PhotonicCircuit
from ..circuit.photonic_compiler import CompiledPhotonicCircuit
from ..results import SimulationResult
from ..state import State
from ..utils import (
    add_heralds_to_state,
    fock_basis,
)
from .task import Task, TaskData
from .task_utils import _check_photon_numbers, _validate_states


class Simulator(Task):
    """
    Simulates a circuit for a provided number of inputs and outputs, returning
    the probability amplitude between them.

    Args:

        circuit (PhotonicCircuit) : The circuit which is to be used for
            simulation.

        inputs (list) : A list of the input states to simulate. For multiple
            inputs this should be a list of States.

        outputs (list | None, optional) : A list of the output states to
            simulate, this can also be set to None to automatically find all
            possible outputs.

    """

    __compatible_backends__ = ("permanent",)

    def __init__(
        self,
        circuit: PhotonicCircuit,
        inputs: State | list[State],
        outputs: State | list[State] | None = None,
    ) -> None:
        # Assign circuit to attribute
        self.circuit = circuit
        self.inputs = inputs  # type: ignore[assignment]
        self.outputs = outputs  # type: ignore[assignment]

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
    def inputs(self) -> list[State]:
        """Stores a lost of all inputs to the system to simulate."""
        return self.__inputs

    @inputs.setter
    def inputs(self, value: State | list[State]) -> None:
        self.__inputs = _validate_states(value, self.circuit.input_modes)

    @property
    def outputs(self) -> list[State] | None:
        """
        A list of all target outputs or None, if None all possible outputs are
        calculated on execution.
        """
        return self.__outputs

    @outputs.setter
    def outputs(self, value: State | list[State] | None) -> None:
        if value is not None:
            value = _validate_states(value, self.circuit.input_modes)
        self.__outputs = value

    def _run(self, backend: FockBackend) -> SimulationResult:  # type: ignore[override]
        """
        Function to run a simulation for a number of inputs/outputs, if no
        outputs are specified then all possible outputs for the photon number
        are calculated. All inputs and outputs should have the same photon
        number.

        Args:

            backend (FockBackend) : The target backend to run the task with.

        Returns:

            SimulationResult : A dictionary containing the calculated
                probability amplitudes, where the first index of the array
                corresponds to the input state, as well as the input and output
                state used to create the array.

        """
        circuit = self.circuit._build()
        if self.outputs is None:
            _check_photon_numbers(self.inputs)
            outputs = fock_basis(
                self.circuit.input_modes, self.inputs[0].n_photons
            )
            outputs = [State(s) for s in outputs]
        else:
            _check_photon_numbers(self.inputs + self.outputs)
            outputs = self.outputs
        in_heralds = self.circuit.heralds["input"]
        out_heralds = self.circuit.heralds["output"]
        # Pre-add output values to avoid doing this many times
        full_outputs = [
            add_heralds_to_state(outs, out_heralds) + [0] * circuit.loss_modes
            for outs in outputs
        ]
        # Calculate permanent for the given inputs and outputs and return
        # values
        amplitudes = np.zeros((len(self.inputs), len(outputs)), dtype=complex)
        for i, ins in enumerate(self.inputs):
            in_state = add_heralds_to_state(ins, in_heralds)
            in_state += [0] * circuit.loss_modes
            for j, outs in enumerate(full_outputs):
                amplitudes[i, j] = backend.probability_amplitude(
                    circuit.U_full, in_state, outs
                )
        # Return results and corresponding states as dictionary
        return SimulationResult(
            amplitudes,
            "probability_amplitude",
            inputs=self.inputs,
            outputs=outputs,
        )

    def _generate_task(self) -> TaskData:
        return SimulatorTask(
            circuit=self.circuit._build(),
            inputs=self.inputs,
            outputs=self.outputs,
        )


@dataclass
class SimulatorTask(TaskData):  # noqa: D101
    circuit: CompiledPhotonicCircuit
    inputs: list[State]
    outputs: list[State] | None
