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


import numpy as np

from ...emulator.backends.fock_backend import FockBackend
from ..circuit import Circuit
from ..results import SimulationResult
from ..state import State
from ..utils import (
    ModeMismatchError,
    PhotonNumberError,
    add_heralds_to_state,
    fock_basis,
)
from .task import Task


class Simulator(Task):
    """
    Simulates a circuit for a provided number of inputs and outputs, returning
    the probability amplitude between them.

    Args:

        circuit : The circuit which is to be used for simulation.

        inputs (list) : A list of the input states to simulate. For multiple
            inputs this should be a list of States.

        outputs (list | None, optional) : A list of the output states to
            simulate, this can also be set to None to automatically find all
            possible outputs.

    """

    __compatible_backends__ = ("permanent",)

    def __init__(
        self,
        circuit: Circuit,
        inputs: State | list[State],
        outputs: State | list[State] | None = None,
    ) -> None:
        # Assign circuit to attribute
        self.circuit = circuit
        self.inputs = inputs  # type: ignore[assignment]
        self.outputs = outputs  # type: ignore[assignment]

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
        # Calculate permanent for the given inputs and outputs and return
        # values
        amplitudes = np.zeros((len(self.inputs), len(outputs)), dtype=complex)
        for i, ins in enumerate(self.inputs):
            in_state = add_heralds_to_state(ins, in_heralds)
            in_state += [0] * circuit.loss_modes
            for j, outs in enumerate(outputs):
                out_state = add_heralds_to_state(outs, out_heralds)
                out_state += [0] * circuit.loss_modes
                amplitudes[i, j] = backend.probability_amplitude(
                    circuit.U_full, in_state, out_state
                )
        # Return results and corresponding states as dictionary
        return SimulationResult(
            amplitudes,
            "probability_amplitude",
            inputs=self.inputs,
            outputs=outputs,
        )


def _validate_states(inputs: State | list[State], n_modes: int) -> list[State]:
    """
    Performs all required processing/checking on the input/outputs states.
    """
    # Convert state to list of States if not provided for single state case
    if isinstance(inputs, State):
        inputs = [inputs]
    # Check each input
    for state in inputs:
        # Ensure correct type
        if not isinstance(state, State):
            raise TypeError(
                "inputs should be a State or list of State objects."
            )
        # Dimension check
        if len(state) != n_modes:
            msg = (
                "One or more input states have an incorrect number of "
                f"modes, correct number of modes is {n_modes}."
            )
            raise ModeMismatchError(msg)
        # Also validate state values
        state._validate()
    return inputs


def _check_photon_numbers(states: list[State]) -> None:
    """
    Raises an exception if photon numbers are mixed when running a
    simulation.
    """
    ns = [s.n_photons for s in states]
    if min(ns) != max(ns):
        raise PhotonNumberError(
            "Mismatch in photon numbers between some inputs/outputs, "
            "this is not currently supported in the Simulator."
        )
