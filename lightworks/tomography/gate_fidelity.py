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

from .mappings import RHO_MAPPING
from .process_tomography import ProcessTomography
from .utils import _calculate_density_matrix, combine_all, state_fidelity

TOMO_INPUTS = ["Z+", "Z-", "X+", "X-", "Y+", "Y-"]


class GateFidelity(ProcessTomography):
    """
    Computers the average gate fidelity

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
        rhos_exp = {}
        for indices in all_inputs:
            index = indices.split(",")
            rho_e = RHO_MAPPING[index[0]]
            for i in index[1:]:
                rho_e = np.kron(rho_e, RHO_MAPPING[i])
            rhos_exp[indices] = (
                np.conj(target_process).T @ rho_e @ target_process
            )
        return np.mean(
            [state_fidelity(rhos[i], rhos_exp[i]) for i in all_inputs],
        ).item()
