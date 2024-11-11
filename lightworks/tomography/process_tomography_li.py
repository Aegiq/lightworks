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

from .mappings import PAULI_MAPPING, RHO_MAPPING
from .process_tomography import ProcessTomography
from .state_tomography import StateTomography as StateTomo
from .utils import combine_all

TOMO_INPUTS = ["Z+", "Z-", "X+", "Y+"]


class LIProcessTomography(ProcessTomography):
    """
    Runs quantum process tomography using the linear inversion estimation
    method.

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
        full_paulis = combine_all(PAULI_MAPPING, self.n_qubits)
        full_rhos = combine_all(RHO_MAPPING, self.n_qubits)
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
