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

from types import FunctionType
from typing import Callable

import numpy as np

from ..sdk.circuit import Circuit
from .mappings import MEASUREMENT_MAPPING, PAULI_MAPPING
from .utils import (
    _calculate_expectation_value,
    _get_required_tomo_measurements,
    _get_tomo_measurements,
    state_fidelity,
)


class StateTomography:
    """
    Generates the required circuit and performs data processing for the
    calculation of the density matrix of a state.

    Args:

        n_qubits (int) : The number of qubits that will be used as part of the
            tomography.

        base_circuit (Circuit) : An initial circuit which produces the required
            output state and can be modified for performing tomography. It is
            required that the number of circuit input modes equals 2 * the
            number of qubits.

        experiment (Callable) : A function for performing the required
            tomography experiments. This should accept a list of circuits and
            return a list of results to process.

        experiment_args (list | None) : Optionally provide additional arguments
            which will be passed directly to the experiment function.

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
        self._rho: np.ndarray

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
    def rho(self) -> np.ndarray:
        """
        The most recently calculated density matrix.
        """
        if not hasattr(self, "_rho"):
            raise AttributeError(
                "Density matrix has not yet been calculated, this can be "
                "achieved with the process method."
            )
        return self._rho

    def process(self) -> np.ndarray:
        """
        Performs the state tomography process with the configured elements to
        calculate the density matrix of the output state.

        Returns:

            np.ndarray : The calculated density matrix from the state tomography
                process.

        """
        req_measurements, result_mapping = _get_required_tomo_measurements(
            self.n_qubits
        )

        # Generate all circuits and run experiment
        circuits = [
            self._create_circuit(
                [MEASUREMENT_MAPPING[g] for g in gates.split(",")]
            )
            for gates in req_measurements
        ]
        all_results = self.experiment(
            circuits,
            *(self.experiment_args if self.experiment_args is not None else []),
        )

        # Convert results into dictionary and then mapping to full set of
        # measurements
        results_dict = dict(zip(req_measurements, all_results))
        results_dict = {
            c: results_dict[result_mapping[c]]
            for c in _get_tomo_measurements(self.n_qubits)
        }

        self._rho = self._calculate_density_matrix(results_dict)
        return self.rho

    def fidelity(self, rho_exp: np.ndarray) -> float:
        """
        Calculates the fidelity of the calculated quantum state against the
        expected density matrix for the state.

        Args:

            rho_exp (np.ndarray) : The expected density matrix.

        Returns:

            float : The calculated fidelity value.

        """
        return state_fidelity(self.rho, rho_exp)

    def _create_circuit(self, measurement_operators: list[Circuit]) -> Circuit:
        """
        Creates a copy of the assigned base circuit and applies the list of
        measurement circuits to each pair of dual-rail encoded qubits.

        Args:

            measurement_operators (list) : A list of 2 mode circuits which act
                as measurement operators to apply to the system.

        Returns:

            Circuit : A modified copy of the base circuit with required
                operations.

        """
        circuit = self.base_circuit.copy()
        # Check number of circuits is correct
        if len(measurement_operators) != self.n_qubits:
            msg = (
                "Number of operators should match number of qubits "
                f"({self.n_qubits})."
            )
            raise ValueError(msg)
        # Add each and then return
        for i, op in enumerate(measurement_operators):
            circuit.add(op, 2 * i)

        return circuit

    def _calculate_density_matrix(self, results: dict[str, dict]) -> np.ndarray:
        """
        Calculates the density matrix using a provided dictionary of results
        data.

        Args:

            results (Result | dict) : Result containing measured output states
                and counts. The keys of this dictionary should be the
                measurement basis and the values should be the results. Each
                result can be one of the returned Result objects used within
                lightworks, or alternatively may just be a dictionary.

        Returns:

            np.ndarray : The calculated density matrix.


        """
        # Process results to find density matrix
        rho = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        for measurement, result in results.items():
            expectation = _calculate_expectation_value(measurement, result)
            expectation /= 2**self.n_qubits
            # Calculate tensor product of the operators used
            ops = measurement.split(",")
            mat = PAULI_MAPPING[ops[0]]
            for g in ops[1:]:
                mat = np.kron(mat, PAULI_MAPPING[g])
            # Updated density matrix
            rho += expectation * mat
        return rho
