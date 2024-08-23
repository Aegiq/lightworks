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

from .. import qubit
from ..sdk.circuit import Circuit
from ..sdk.state import State
from .utils import state_fidelity

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
        req_measurements, result_mapping = (
            StateTomography._get_required_measurements(self.n_qubits)
        )

        # Generate all circuits and run experiment
        circuits = [
            self._create_circuit(
                [StateTomography._get_measurement_operator(g) for g in gates]
            )
            for gates in req_measurements
        ]
        args = self.experiment_args if self.experiment_args is not None else []
        all_results = self.experiment(circuits, *args)

        # Convert results into dictionary and then mapping to full set of
        # measurements
        results_dict = dict(zip(req_measurements, all_results))
        results_dict = {
            c: results_dict[result_mapping[c]]
            for c in StateTomography._get_all_measurements(self.n_qubits)
        }

        self._rho = StateTomography._calculate_density_matrix(
            self.n_qubits, results_dict
        )
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

    @staticmethod
    def _calculate_density_matrix(
        n_qubits: int, results: dict[str, dict]
    ) -> np.ndarray:
        """
        Calculates the density matrix using a provided dictionary of results
        data. The keys of this dictionary should be the measurement basis and
        the values the results.
        """
        # Process results to find density matrix
        rho = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        for comb, res in results.items():
            gates = [*comb]
            total = 0
            n_counts = 0
            for s, c in res.items():
                n_counts += c
                # Adjust multiplier to account for variation in eigenvalues
                multiplier = 1
                for j, gate in enumerate(gates):
                    if gate == "I" or s[2 * j : 2 * j + 2] == State([1, 0]):
                        multiplier *= 1
                    elif s[2 * j : 2 * j + 2] == State([0, 1]):
                        multiplier *= -1
                total += multiplier * c
            total /= (2**n_qubits) * n_counts
            # Calculate tensor product of the operators used
            mat = StateTomography._get_pauli_matrix(gates[0])
            for g in gates[1:]:
                mat = np.kron(mat, StateTomography._get_pauli_matrix(g))
            # Updated density matrix
            rho += total * mat
        return rho

    @staticmethod
    def _get_all_measurements(n_qubits: int) -> list[str]:
        """
        Returns all measurements required for a state tomography of n qubits.
        """
        # Find all measurement combinations
        measurements = list(MEASUREMENT_MAPPING.keys())
        for _i in range(n_qubits - 1):
            measurements = [
                g1 + g2 for g1 in measurements for g2 in MEASUREMENT_MAPPING
            ]
        return measurements

    @staticmethod
    def _get_required_measurements(n_qubits: int) -> tuple[list, dict]:
        """
        Calculates reduced list of required measurements assuming that any
        measurements in the I basis can be replaced with a Z measurement.
        A dictionary which maps the full measurements to the reduced basis is
        also returned.
        """
        mapping = {
            c: c.replace("I", "Z")
            for c in StateTomography._get_all_measurements(n_qubits)
        }
        req_measurements = list(set(mapping.values()))
        return req_measurements, mapping

    @staticmethod
    def _get_measurement_operator(measurement: str) -> Circuit:
        """
        Returns the circuit required to transform between a measurement into the
        Z basis.
        """
        if measurement not in MEASUREMENT_MAPPING:
            raise ValueError("Provided measurement value not recognised.")
        return MEASUREMENT_MAPPING[measurement]

    @staticmethod
    def _get_pauli_matrix(measurement: str) -> np.ndarray:
        """
        Returns the pauli matrix associated with an observable.
        """
        if measurement not in PAULI_MAPPING:
            raise ValueError("Provided measurement value not recognised.")
        return PAULI_MAPPING[measurement]
