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

from ..sdk.circuit import Circuit
from ..sdk.state import State
from .process_tomography import ProcessTomography
from .state_tomography import StateTomography as StateTomo
from .utils import INPUT_MAPPING, PAULI_MAPPING, RHO_MAPPING

TOMO_INPUTS = ["Z+", "Z-", "X+", "Y+"]


class LIProcessTomography(ProcessTomography):
    """
    Generates the required configurations for the calculation of the choi matrix
    representation of a process.

    Args:

        n_qubits (int) : The number of qubits that will be used as part of the
            tomography.

        base_circuit (Circuit) : An initial circuit which produces the required
            output state and can be modified for performing tomography. It is
            required that the number of circuit input modes equals 2 * the
            number of qubits.

        experiment (Callable) : A function for performing the required
            tomography experiments. This should accept a list of circuits and a
            list of inputs and then return a list of results to process.

        experiment_args (list | None) : Optionally provide additional arguments
            which will be passed directly to the experiment function.

    """

    def process(self) -> np.ndarray:
        """
        Performs process tomography with the configured elements and calculates
        the choi matrix using linear inversion.

        Returns:

            np.ndarray : The calculated choi matrix for the process.

        """
        all_inputs = list(TOMO_INPUTS)
        for _ in range(self.n_qubits - 1):
            all_inputs = [
                i1 + "," + i2 for i1 in all_inputs for i2 in TOMO_INPUTS
            ]
        results = self._run_required_experiments(all_inputs)
        # Get expectation values using results
        lambdas = self._calculate_expectation_values(results)
        # Find all pauli and density matrices for multi-qubit states
        full_paulis = dict(PAULI_MAPPING)
        full_rhos = dict(RHO_MAPPING)
        for _ in range(self.n_qubits - 1):
            full_paulis = {
                k1 + "," + k2: np.kron(v1, v2)
                for k1, v1 in full_paulis.items()
                for k2, v2 in PAULI_MAPPING.items()
            }
            full_rhos = {
                k1 + "," + k2: np.kron(v1, v2)
                for k1, v1 in full_rhos.items()
                for k2, v2 in RHO_MAPPING.items()
            }
        # Determine the transformation matrix to perform linear inversion
        dim = 2**self.n_qubits
        transform_matrix = np.zeros((dim**4, dim**4), dtype=complex)
        for i, (in_s, meas) in enumerate(lambdas):
            transform_matrix[i, :] = (
                np.kron(np.array(full_rhos[in_s]).conj(), full_paulis[meas])
                .flatten()
                .conj()
            )
        # Then find the choi matrix
        choi = np.linalg.pinv(transform_matrix) @ np.array(
            list(lambdas.values())
        )
        self._choi = choi.reshape(dim**2, dim**2)
        return self.choi

    def _calculate_expectation_values(
        self, results: dict[tuple[str, str], dict]
    ) -> dict[tuple[str, str], float]:
        """
        Calculates the expectation values from a set of results containing,
        input states, observables and measurement data for each.
        """
        return {
            (in_state, measurement): StateTomo._calculate_expectation_value(
                measurement, res
            )
            for (in_state, measurement), res in results.items()
        }

    def _run_required_experiments(
        self, inputs: list[str]
    ) -> dict[tuple[str, str], dict[str, int]]:
        """
        Runs all required experiments to find density matrices for a provided
        set of inputs.
        """
        req_measurements, result_mapping = StateTomo._get_required_measurements(
            self.n_qubits
        )
        # Determine required input states and circuits
        all_circuits = []
        all_input_states = []
        for in_state in inputs:
            for meas in req_measurements:
                circ, state = self._create_circuit_and_input(in_state, meas)
                all_circuits.append(circ)
                all_input_states.append(state)
        # Run all required experiments
        results = self.experiment(
            all_circuits,
            all_input_states,
            *(self.experiment_args if self.experiment_args is not None else []),
        )
        # Sort results into each input/measurement combination
        num_per_in = len(req_measurements)
        sorted_results = {
            in_state: dict(
                zip(
                    req_measurements,
                    results[num_per_in * i : num_per_in * (i + 1)],
                )
            )
            for i, in_state in enumerate(inputs)
        }
        # Expand results to include all of the required measurements
        full_results = {}
        for in_state, res in sorted_results.items():
            for meas in StateTomo._get_all_measurements(self.n_qubits):
                full_results[in_state, meas] = res[result_mapping[meas]]
        return full_results

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
            circ.add(StateTomo._get_measurement_operator(op), 2 * i)
        return circ, in_state
