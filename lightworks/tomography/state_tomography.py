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
from scipy.linalg import sqrtm

from .. import qubit
from ..sdk.circuit import Circuit
from ..sdk.state import State

_y_measure = Circuit(2)
_y_measure.add(qubit.S())
_y_measure.add(qubit.Z())
_y_measure.add(qubit.H())

MEASUREMENT_MAPPING = {
    "I": qubit.I(),
    "X": qubit.H(),
    "Y": _y_measure,
    "Z": qubit.I(),
}

PAULI_MAPPING = {
    "I": np.array([[1, 0], [0, 1]]),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}


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

    """

    def __init__(
        self, n_qubits: int, base_circuit: Circuit, experiment: Callable
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
        # Find measurement combinations
        combinations = list(MEASUREMENT_MAPPING.keys())
        for _i in range(self.n_qubits - 1):
            combinations = [
                g1 + g2 for g1 in combinations for g2 in MEASUREMENT_MAPPING
            ]

        # Generate all circuits and run experiment
        circuits = [
            self._create_circuit(
                [self._get_measurement_operator(g) for g in gates]
            )
            for gates in combinations
        ]
        all_results = self.experiment(circuits)

        # Process results to find density matrix
        rho = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        for i, gates in enumerate([[*c] for c in combinations]):
            results = all_results[i]
            total = 0
            n_counts = 0
            for s, c in results.items():
                n_counts += c
                # Adjust multiplier to account for variation in eigenvalues
                multiplier = 1
                for j, gate in enumerate(gates):
                    if gate == "I" or s[2 * j : 2 * j + 2] == State([1, 0]):
                        multiplier *= 1
                    elif s[2 * j : 2 * j + 2] == State([0, 1]):
                        multiplier *= -1
                total += multiplier * c
            total /= (2**self.n_qubits) * n_counts
            # Calculate tensor product of the operators used
            mat = self._get_pauli_matrix(gates[0])
            for g in gates[1:]:
                mat = np.kron(mat, self._get_pauli_matrix(g))
            # Updated density matrix
            rho += total * mat

        # Assign to attribute then return
        self._rho = rho
        return self._rho

    def fidelity(self, rho_exp: np.ndarray) -> float:
        """
        Calculates the fidelity of the quantum state against the expected
        density matrix for the state.

        Args:

            rho_exp (np.ndarray) : The expected density matrix.

        Returns:

            float : The calculated fidelity value.

        """
        rho_exp = np.array(rho_exp)
        rho_root = sqrtm(self._rho)
        inner = rho_root @ rho_exp @ rho_root
        return abs(np.trace(sqrtm(inner)))

    def _create_circuit(self, measurement_operators: list) -> Circuit:
        """
        Creates a copy of the assigned base circuit and applies the list of
        measurement circuits to each pair of dual-rail encoded qubits.
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

    def _get_measurement_operator(self, measurement: str) -> Circuit:
        """
        Returns the circuit required to transform between a measurement into the
        Z basis.
        """
        if measurement not in MEASUREMENT_MAPPING:
            raise ValueError("Provided measurement value not recognised.")
        return MEASUREMENT_MAPPING[measurement]

    def _get_pauli_matrix(self, measurement: str) -> np.ndarray:
        """
        Returns the pauli matrix associated with an observable.
        """
        if measurement not in PAULI_MAPPING:
            raise ValueError("Provided measurement value not recognised.")
        return PAULI_MAPPING[measurement]
