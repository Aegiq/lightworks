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


from ..circuit import PhotonicCircuit
from ..state import State
from ..utils import validate_states
from .data import SimulatorTask
from .task import Task


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
        self.__inputs = validate_states(value, self.circuit.input_modes)

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
            value = validate_states(value, self.circuit.input_modes)
        self.__outputs = value

    def _generate_task(self) -> SimulatorTask:
        return SimulatorTask(
            circuit=self.circuit._build(),
            inputs=self.inputs,
            outputs=self.outputs,
        )
