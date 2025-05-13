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
from numpy.typing import NDArray

from lightworks.sdk.state import State

from .mappings import PAULI_MAPPING, RHO_MAPPING
from .process_tomography import _ProcessTomography
from .utils import (
    _calculate_expectation_value,
    _combine_all,
    _unvec,
    _vec,
    process_fidelity,
)


class LIProcessTomography(_ProcessTomography):
    """
    Runs quantum process tomography using the linear inversion estimation
    method.

    Args:

        n_qubits (int) : The number of qubits that will be used as part of the
            tomography.

        base_circuit (PhotonicCircuit) : An initial circuit which realises the
            required operation and can be modified for performing tomography.
            It is required that the number of circuit input modes equals 2 * the
            number of qubits.

    """

    @property
    def choi(self) -> NDArray[np.complex128]:
        """Returns the calculate choi matrix for a circuit."""
        if not hasattr(self, "_choi"):
            raise AttributeError(
                "Choi matrix has not yet been calculated, this can be achieved "
                "with the process method."
            )
        return self._choi

    def process(
        self,
        data: list[dict[State, int]] | dict[tuple[str, str], dict[State, int]],
    ) -> NDArray[np.complex128]:
        """
        Performs process tomography with the configured elements and calculates
        the choi matrix using linear inversion.

        Args:

            data (list | dict) : The collected measurement data. If a list then
                this should match the order the experiments were provided, and
                if a dictionary, then each key should be tuple of the input and
                measurement basis.

        Returns:

            np.ndarray : The calculated choi matrix for the process.

        """
        results = self._convert_tomography_data(data)
        # Get expectation values using results
        lambdas = self._calculate_expectation_values(results)
        # Find all pauli and density matrices for multi-qubit states
        full_paulis = _combine_all(PAULI_MAPPING, self.n_qubits)
        full_rhos = _combine_all(RHO_MAPPING, self.n_qubits)
        # Determine the transformation matrix to perform linear inversion
        dim = 2**self.n_qubits
        transform_matrix = np.zeros((dim**4, dim**4), dtype=complex)
        for i, (in_s, meas) in enumerate(lambdas):
            transform_matrix[i, :] = _vec(
                np.kron(np.array(full_rhos[in_s]).conj(), full_paulis[meas])
            ).conj()
        # Then find the choi matrix
        choi = np.linalg.pinv(transform_matrix) @ np.array(
            list(lambdas.values())
        )
        self._choi = _unvec(choi)
        return self.choi

    def fidelity(self, choi_exp: NDArray[np.complex128]) -> float:
        """
        Calculates fidelity of the calculated choi matrix compared to the
        expected one.
        """
        return process_fidelity(self.choi, choi_exp)

    def _calculate_expectation_values(
        self, results: dict[tuple[str, str], dict[State, int]]
    ) -> dict[tuple[str, str], float]:
        """
        Calculates the expectation values from a set of results containing,
        input states, observables and measurement data for each.
        """
        return {
            (in_state, measurement): _calculate_expectation_value(
                measurement, res
            )
            for (in_state, measurement), res in results.items()
        }
