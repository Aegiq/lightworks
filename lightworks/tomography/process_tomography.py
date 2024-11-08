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

from abc import ABCMeta, abstractmethod
from types import FunctionType
from typing import Callable

import numpy as np

from ..sdk.circuit import Circuit
from ..sdk.state import State
from .mappings import INPUT_MAPPING, MEASUREMENT_MAPPING
from .utils import process_fidelity

TOMO_INPUTS = ["Z+", "Z-", "X+", "Y+"]


class ProcessTomography(metaclass=ABCMeta):
    """
    Process tomography base class.
    """

    def __init__(
        self,
        n_qubits: int,
        base_circuit: Circuit,
        experiment: Callable,
        experiment_args: list | None = None,
    ) -> None:
        # Type check inputs
        if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
            raise TypeError("Number of qubits should be an integer.")
        if not isinstance(base_circuit, Circuit):
            raise TypeError("Base circuit should be a circuit object.")

        if 2 * n_qubits != base_circuit.input_modes:
            msg = (
                "Number of circuit input modes does not match the amount "
                "required for the specified number of qubits, expected "
                f"{2 * n_qubits}."
            )
            raise ValueError(msg)

        self._n_qubits = n_qubits
        self._base_circuit = base_circuit
        self.experiment = experiment
        self.experiment_args = experiment_args
        self._choi: np.ndarray

    @property
    def base_circuit(self) -> Circuit:
        """
        The base circuit which is to be modified as part of the tomography
        calculations.
        """
        return self._base_circuit

    @property
    def n_qubits(self) -> int:
        """
        The number of qubits within the system.
        """
        return self._n_qubits

    @property
    def experiment(self) -> Callable:
        """
        A function to call which runs the required experiments. This should
        accept a list of circuits as a single argument and then return a list
        of the corresponding results, with each result being a dictionary or
        Results object containing output states and counts.
        """
        return self._experiment

    @experiment.setter
    def experiment(self, value: Callable) -> None:
        if not isinstance(value, FunctionType):
            raise TypeError(
                "Provided experiment should be a function which accepts a list "
                "of circuits and returns a list of results containing only the "
                "qubit modes."
            )
        self._experiment = value

    @property
    def choi(self) -> np.ndarray:
        """Returns the calculate choi matrix for a circuit."""
        if not hasattr(self, "_choi"):
            raise AttributeError(
                "Choi matrix has not yet been calculated, this can be achieved "
                "with the process method."
            )
        return self._choi

    @abstractmethod
    def process(self) -> np.ndarray:
        """
        Performs tomography using the selected algorithm.
        """

    def fidelity(self, choi_exp: np.ndarray) -> float:
        """
        Calculates fidelity of the calculated choi matrix compared to the
        expected one.
        """
        return process_fidelity(self.choi, choi_exp)

    def _create_circuit_and_input(
        self, input_op: str, output_op: str
    ) -> tuple[Circuit, State]:
        """
        Creates the required circuit and input state to achieve a provided input
        and output operation.
        """
        in_state = State([])
        circ = Circuit(self.base_circuit.input_modes)
        # Input operation
        for i, op in enumerate(input_op.split(",")):
            in_state += INPUT_MAPPING[op][0]
            circ.add(INPUT_MAPPING[op][1], 2 * i)
        # Add base circuit
        circ.add(self.base_circuit)
        # Measurement operation
        for i, op in enumerate(output_op.split(",")):
            circ.add(MEASUREMENT_MAPPING[op], 2 * i)
        return circ, in_state
