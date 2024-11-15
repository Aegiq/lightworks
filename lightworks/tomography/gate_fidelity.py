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
from .utils import _calculate_density_matrix, combine_all, vec

TOMO_INPUTS = ["Z+", "Z-", "X+", "Y+"]


class GateFidelity(ProcessTomography):
    """
    Computes the average gate fidelity using equation 19 from
    https://arxiv.org/pdf/quant-ph/0205035, which involves performing state
    tomography for a set of inputs.

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

    @property
    def choi(self) -> None:
        """
        Overwrites default behaviour for return of choi matrix.
        """
        raise AttributeError("Gate fidelity does not computer choi matrix")

    def process(self, target_process: np.ndarray) -> float:
        """
        Calculates gate fidelity for an expected target process

        Args:

            target_process (np.ndarray) : The unitary matrix corresponding to
                the target process. The dimension of this should be 2^n_qubits.

        Returns:

            float : The calculated fidelity.

        """
        all_inputs = combine_all(TOMO_INPUTS, self.n_qubits)
        results = self._run_required_experiments(all_inputs)
        # Sorted results per input
        remapped_results = {k[0]: {} for k in results}
        for (k1, k2), r in results.items():
            remapped_results[k1][k2] = r
        # Calculate density matrices
        rhos = {
            k: _calculate_density_matrix(v, self.n_qubits)
            for k, v in remapped_results.items()
        }
        # Unitary basis
        u_basis = combine_all(dict(PAULI_MAPPING), self.n_qubits)
        # Input density matrices
        rho_basis = combine_all(
            {i: RHO_MAPPING[i] for i in TOMO_INPUTS}, self.n_qubits
        )
        # Convert both rho and rho_basis into vector to ensure order matches
        rho_vec = [rhos[i] for i in all_inputs]
        rho_basis_vec = [rho_basis[i] for i in all_inputs]

        alpha_mat = self._calculate_alpha(u_basis.values(), rho_basis_vec)

        total = 0
        for i, uj in enumerate(u_basis.values()):
            eps_uj = sum(
                alpha_mat[i][j] * rho_vec[j] for j in range(len(rho_vec))
            )
            inner = (
                target_process
                @ np.conj(uj.T)
                @ np.conj(target_process.T)
                @ eps_uj
            )
            total += np.trace(inner)
        dim = 2**self.n_qubits
        return (total + dim**2) / (dim**2 * (dim + 1))

    def _calculate_alpha(
        self, u_basis: list[np.ndarray], rho_basis: list[np.ndarray]
    ) -> np.ndarray:
        """
        Solves for provided matrix in terms of basis.
        """
        alpha = np.zeros(
            (2 ** (2 * self.n_qubits), 2 ** (2 * self.n_qubits)), dtype=complex
        )
        basis_vectors = np.column_stack([vec(m) for m in rho_basis])
        for i, u in enumerate(u_basis):
            alpha[i, :] = np.linalg.solve(basis_vectors, vec(u))

        return alpha
