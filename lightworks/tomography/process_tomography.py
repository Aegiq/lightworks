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

from typing import Callable

import numpy as np

from .. import qubit
from ..sdk.circuit import Circuit
from ..sdk.state import State
from .state_tomography import StateTomography as StateTomo
from .utils import process_fidelity

TOMO_INPUTS = ["0", "1", "+", "R"]

r_transform = qubit.H()
r_transform.add(qubit.S())

INPUT_MAPPING: dict[str, tuple[State, Circuit]] = {
    "0": (State([1, 0]), qubit.I()),
    "1": (State([0, 1]), qubit.I()),
    "+": (State([1, 0]), qubit.H()),
    "R": (State([1, 0]), r_transform),
}
BASIS_MAPPING: dict[str, list] = {
    "0": [1, 0],
    "1": [0, 1],
    "+": [2**-0.5, 2**-0.5],
    "R": [2**-0.5, 1j * 2**-0.5],
}


class ProcessTomography:
    """
    Desc
    """

    def __init__(
        self,
        n_qubits: int,
        base_circuit: Circuit,
        experiment: Callable,
        experiment_args: list | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.base_circuit = base_circuit
        self.experiment = experiment
        self.experiment_args = experiment_args

    @property
    def chi(self) -> np.ndarray:
        """Returns the calculate chi matrix for a circuit."""
        if not hasattr(self, "_chi"):
            raise AttributeError(
                "Chi has not yet been calculated, this can be achieved with the"
                "process method."
            )
        return self._chi

    def process(self) -> np.ndarray:
        """
        Desc
        """
        all_inputs = list(TOMO_INPUTS)
        for _ in range(self.n_qubits - 1):
            all_inputs = [i1 + i2 for i1 in all_inputs for i2 in TOMO_INPUTS]

        results = self._run_required_experiments(all_inputs)

        rhos_full = self._calculate_density_matrices(results)
        rhos = [rhos_full[in_s] for in_s in all_inputs]

        transform = self._calculate_transform_matrix(self.n_qubits, all_inputs)

        rhos_t = [
            sum(transform[i][j] * rhos[j] for j in range(len(rhos)))
            for i in range(len(rhos))
        ]
        full_mat = np.zeros(
            (2 ** (2 * self.n_qubits), 2 ** (2 * self.n_qubits)), dtype=complex
        )

        for i in range(2**self.n_qubits):
            for j in range(2**self.n_qubits):
                mat = rhos_t[2**self.n_qubits * i + j]
                full_mat[
                    2**self.n_qubits * i : 2**self.n_qubits * (i + 1),
                    2**self.n_qubits * j : 2**self.n_qubits * (j + 1),
                ] = mat[:, :]

        x_mat = np.array([[0, 1], [1, 0]])
        a_mat = 0.5 * (
            np.kron(np.array([[1, 0], [0, -1]]), np.identity(2))
            + np.kron(x_mat, x_mat)
        )
        if self.n_qubits == 1:
            k_mat = a_mat
        elif self.n_qubits == 2:
            m_mat = np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
            )
            a2_mat = np.kron(a_mat, a_mat)
            p_mat = np.kron(np.identity(2), np.kron(m_mat, np.identity(2)))
            k_mat = p_mat @ a2_mat
        else:
            raise NotImplementedError("Not yet generalized.")

        self._chi = k_mat.T @ full_mat @ k_mat

        return self.chi

    def fidelity(self, chi_exp: np.ndarray) -> float:
        """
        Calculates fidelity of the calculated chi matrix compared to the
        expected one.
        """
        return process_fidelity(self.chi, chi_exp)

    def _run_required_experiments(
        self, inputs: list[str]
    ) -> dict[str, dict[str, dict]]:
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
            full_res = {
                meas: res[result_mapping[meas]]
                for meas in StateTomo._get_all_measurements(self.n_qubits)
            }
            full_results[in_state] = full_res
        return full_results

    def _calculate_density_matrices(
        self, results: dict[str, dict[str, dict]]
    ) -> dict[str, np.ndarray]:
        """
        Calculates density matrices for each input and set of results in the
        provided dictionary.
        """
        return {
            in_state: StateTomo._calculate_density_matrix(self.n_qubits, res)
            for in_state, res in results.items()
        }

    def _calculate_transform_matrix(
        self, n_qubits: int, inputs: list[str]
    ) -> np.ndarray:
        """
        Calculates the matrix required to transform between input basis and
        pauli basis.
        """
        basis = []
        for in_state in inputs:
            in_vector = np.array(BASIS_MAPPING[in_state[0]], dtype=complex)
            for i in range(n_qubits - 1):
                in_vector = np.kron(in_vector, BASIS_MAPPING[in_state[i + 1]])
            basis.append(np.outer(in_vector, np.conj(in_vector.T)))
        basis_vectors = np.column_stack([m.flatten() for m in basis])

        transform = np.zeros(
            (2 ** (2 * n_qubits), 2 ** (2 * n_qubits)), dtype=complex
        )
        for i in range(len(inputs)):
            target_rho = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
            target_rho[int(i / 2**n_qubits), i % (2**n_qubits)] = 1

            mu = np.linalg.solve(basis_vectors, target_rho.flatten())
            transform[i, :] = mu[:]

        return transform

    def _create_circuit_and_input(
        self, input_op: str, output_op: str
    ) -> tuple[Circuit, State]:
        """
        Desc
        """
        in_state = State([])
        circ = Circuit(self.base_circuit.input_modes)

        for i, op in enumerate(input_op):
            in_state += INPUT_MAPPING[op][0]
            circ.add(INPUT_MAPPING[op][1], 2 * i)

        circ.add(self.base_circuit)

        for i, op in enumerate(output_op):
            circ.add(StateTomo._get_measurement_operator(op), 2 * i)

        return circ, in_state
