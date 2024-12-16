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

from ..circuit import PhotonicCircuit
from ..state import State
from ..utils import PostSelectionType, process_post_selection, validate_states
from .data import AnalyzerTask
from .task import Task


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
        post_selection: PostSelectionType
        | Callable[[State], bool]
        | None = None,
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
    def post_selection(self) -> PostSelectionType | None:
        """
        Stores post-selection criteria for analysis.
        """
        return self.__post_selection

    @post_selection.setter
    def post_selection(
        self, value: PostSelectionType | Callable[[State], bool] | None
    ) -> None:
        value = process_post_selection(value)
        self.__post_selection = value

    @property
    def inputs(self) -> list[State]:
        """Store list of target inputs to the system."""
        return self.__inputs

    @inputs.setter
    def inputs(self, value: State | list[State]) -> None:
        self.__inputs = validate_states(value, self.circuit.input_modes)

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

    def _generate_task(self) -> AnalyzerTask:
        return AnalyzerTask(
            circuit=self.circuit._build(),
            inputs=self.inputs,
            expected=self.expected,
            post_selection=self.post_selection,
        )
