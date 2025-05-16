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
from .utils import _check_target_process, _combine_all, _vec


class GateFidelity(_ProcessTomography):
    """
    Computes the average gate fidelity using equation 19 from
    https://arxiv.org/pdf/quant-ph/0205035, which involves performing state
    tomography for a set of inputs.

    Args:

        n_qubits (int) : The number of qubits that will be used as part of the
            tomography.

        base_circuit (PhotonicCircuit) : An initial circuit which realises the
            required operation and can be modified for performing tomography.
            It is required that the number of circuit input modes equals 2 * the
            number of qubits.

    """

    @property
    def fidelity(self) -> float:
        """Returns the calculated fidelity of a process."""
        if not hasattr(self, "_fidelity"):
            raise AttributeError(
                "Fidelity has not yet been calculated, this can be achieved "
                "with the process method."
            )
        return self._fidelity

    def process(
        self,
        data: list[dict[State, int]] | dict[tuple[str, str], dict[State, int]],
        target_process: NDArray[np.complex128],
    ) -> float:
        """
        Calculates gate fidelity for an expected target process

        Args:

            data (list | dict) : The collected measurement data. If a list then
                this should match the order the experiments were provided, and
                if a dictionary, then each key should be tuple of the input and
                measurement basis.

            target_process (np.ndarray) : The unitary matrix corresponding to
                the target process. The dimension of this should be 2^n_qubits.

        Returns:

            float : The calculated fidelity.

        """
        target_process = _check_target_process(target_process, self.n_qubits)
        # Get density matrices and convert to vector
        rhos = self.process_density_matrices(data)
        rho_vec = [rhos[i] for i in self._full_input_basis()]
        # Get pauli matrix basis and coefficients relating to unitary
        alpha_mat, u_basis = self._calculate_alpha_and_u_basis()
        # Find sum from equation
        total = 0
        for i, uj in enumerate(u_basis):
            for j, rho in enumerate(rho_vec):
                total += alpha_mat[i][j] * np.trace(
                    target_process
                    @ np.conj(uj.T)
                    @ np.conj(target_process.T)
                    @ rho
                )
        # Use total within calculation and return
        dim = 2**self.n_qubits
        self._fidelity = np.real((total + dim**2) / (dim**2 * (dim + 1))).item()
        return self.fidelity

    def _calculate_alpha_and_u_basis(
        self,
    ) -> tuple[NDArray[np.complex128], list[NDArray[np.complex128]]]:
        """
        Calculates the pauli matrix basis to use for the calculation and finds
        the coefficients of the alpha matrix used which can be used to the
        relate the input density matrices to the pauli matrix basis.
        """
        # Unitary basis
        u_basis = list(
            _combine_all(dict(PAULI_MAPPING), self.n_qubits).values()
        )
        # Input density matrices
        rho_basis = _combine_all(
            [RHO_MAPPING[i] for i in list(self._tomo_inputs)], self.n_qubits
        )
        # Then find alpha matrix
        alpha = np.zeros(
            (2 ** (2 * self.n_qubits), 2 ** (2 * self.n_qubits)), dtype=complex
        )
        basis_vectors = np.column_stack([_vec(m) for m in rho_basis])
        for i, u in enumerate(u_basis):
            alpha[i, :] = np.linalg.solve(basis_vectors, _vec(u))
        return alpha, u_basis
