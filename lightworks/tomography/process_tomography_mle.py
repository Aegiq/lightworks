# Copyright 2024 - 2025 Aegiq Ltd.
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

import warnings

import numpy as np
from numpy.typing import NDArray

from lightworks.sdk.state import State

from .mappings import PAULI_MAPPING, RHO_MAPPING
from .process_tomography import _ProcessTomography
from .projection import _unvec, _vec, project_choi_to_physical
from .utils import (
    _calculate_expectation_value,
    _combine_all,
    _get_tomo_measurements,
    process_fidelity,
)


class MLEProcessTomography(_ProcessTomography):
    """
    Runs quantum process tomography using the maximum likelihood estimation
    method.

    Args:

        n_qubits (int) : The number of qubits that will be used as part of the
            tomography.

        base_circuit (PhotonicCircuit) : An initial circuit which realises the
            required operation and can be modified for performing tomography.
            It is required that the number of circuit input modes equals 2 * the
            number of qubits.

    """

    _tomo_inputs = ("Z+", "Z-", "X+", "X-", "Y+", "Y-")

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
        the choi matrix using maximum likelihood estimation.

        Args:

            data (list | dict) : The collected measurement data. If a list then
                this should match the order the experiments were provided, and
                if a dictionary, then each key should be tuple of the input and
                measurement basis.

        Returns:

            np.ndarray : The calculated choi matrix for the process.

        """
        results = self._convert_tomography_data(data)
        nij = {}
        # Convert counts to an expectation value
        for (in_state, meas), result in results.items():
            # Remove trivial measurement here
            if meas == ",".join("I" * self.n_qubits):
                continue
            nij[in_state, meas] = _calculate_expectation_value(meas, result)
        # Run MLE algorithm and get choi
        mle = MLETomographyAlgorithm(self.n_qubits, self._full_input_basis())
        self._choi = mle.pgdb(nij)
        return self.choi

    def fidelity(self, choi_exp: NDArray[np.complex128]) -> float:
        """
        Calculates fidelity of the calculated choi matrix compared to the
        expected one.
        """
        return process_fidelity(self.choi, choi_exp)


class MLETomographyAlgorithm:
    """
    Implements the pgdB algorithm for maximum likelihood estimation from
    https://arxiv.org/abs/1803.10062 for calculation of a physical choi matrix
    from the tomography measurement data.

    Args:

        n_qubits (int) : The number of qubits used within the tomography.

    """

    def __init__(self, n_qubits: int, input_basis: list[str]) -> None:
        self.n_qubits = n_qubits

        # Pre-calculate all required n_qubit pauli & density matrices + the
        # inputs and measurement basis for the tomography
        self._all_rhos = _combine_all(RHO_MAPPING, self.n_qubits)
        self._all_pauli = _combine_all(PAULI_MAPPING, self.n_qubits)
        self._input_basis = list(input_basis)
        self._meas_basis = _get_tomo_measurements(n_qubits, remove_trivial=True)

        self._a_matrix = self._a_mat()

    def pgdb(
        self,
        data: dict[tuple[str, str], float],
        max_iter: int = 1000,
        stop_threshold: float = 1e-10,
    ) -> NDArray[np.complex128]:
        """
        Runs the pgdB algorithm on the provided data set.

        Args:

            data (dict) : The measured tomography experiment data. The keys of
                this dictionary should be the input/measurement basis and the
                values should be the calculated expectation values.

            max_iter (int, optional) : Sets the maximum number of iterations
                that the algorithm can perform, defaults to 1000.

            stop_threshold (float, optional) : Sets the stopping threshold for
                the gradient descent algorithm. Defaults to 1e-10.

        Returns:

            np.ndarray : The calculated choi matrix

        """
        # Convert data to required vector format
        n_vec = self._n_vec_from_data(data)

        # Initialize choi as identity and find initial cost
        dim = 2**self.n_qubits
        choi = np.identity(dim**2, dtype=complex) / dim
        current_cost = self._cost(choi, n_vec)

        # Define algorithm parameters
        mu = 3 / (2 * dim**2)
        gamma = 0.3

        # Run gradient descent
        for _ in range(max_iter):
            # Find modification to the choi matrix
            mod = project_choi_to_physical(
                choi - 1 / mu * self._gradient(choi, n_vec),
                max_iter=max_iter,
            )
            mod -= choi

            # Optimise alpha weighting parameter
            alpha = 0.5
            new_cost = self._cost(choi + alpha * mod, n_vec)
            thresh_value = gamma * np.trace(
                mod @ np.conj(self._gradient(choi.T, n_vec))
            )
            while new_cost > current_cost + alpha * thresh_value:
                alpha *= 0.5
                new_cost = self._cost(choi + alpha * mod, n_vec)

            # Update choi
            choi += alpha * mod

            # Check cost is still improving, otherwise stop
            if current_cost - new_cost < stop_threshold:
                break
            current_cost = new_cost
        # Warn if the max iterations are exceeded
        else:
            warnings.warn("Max iterations exceeded.", stacklevel=1)

        return choi

    def _n_vec_from_data(
        self, data: dict[tuple[str, str], float]
    ) -> NDArray[np.float64]:
        """
        Converts the data from dictionary format into the required vector,
        ensuring this is correct regardless of the data ordering of the
        dictionary.
        """
        n_vec = np.zeros(2 * len(data), dtype=complex)
        for i, in_s in enumerate(self._input_basis):
            for j, meas in enumerate(self._meas_basis):
                coord = 2 * (len(self._meas_basis) * i + j)
                n = data[in_s, meas]
                n_vec[coord : coord + 2] = [(1 + n) / 2, (1 - n) / 2]
        return n_vec / len(data)

    def _a_mat(self) -> NDArray[np.complex128]:
        """
        Calculates the A matrix which is used for vectorisation of the pgdB
        algorithm.
        """
        dim1 = len(self._input_basis) * len(self._meas_basis) * 2
        dim2 = 2 ** (4 * self.n_qubits)
        a_mat = np.zeros((dim1, dim2), dtype=complex)
        id_mat = np.identity(2**self.n_qubits, dtype=complex)
        for i, in_s in enumerate(self._input_basis):
            for j, meas in enumerate(self._meas_basis):
                obs = self._all_pauli[meas]
                a_mat[2 * (len(self._meas_basis) * i + j), :] = _vec(
                    np.kron(self._all_rhos[in_s], ((id_mat + obs) / 2).T)
                )[:]
                a_mat[2 * (len(self._meas_basis) * i + j) + 1, :] = _vec(
                    np.kron(self._all_rhos[in_s], ((id_mat - obs) / 2).T)
                )[:]
        return a_mat / (2 ** (2 * self.n_qubits))

    def _p_vec(self, choi: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Calculates the expected measurement outcomes from the provided choi
        matrix.
        """
        return (self._a_matrix @ _vec(choi.T)).clip(1e-8)

    def _cost(
        self, choi: NDArray[np.complex128], n_vec: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """
        Computes the variation between the current choi matrix and the
        calculated values.
        """
        return -n_vec.T @ np.log(self._p_vec(choi))

    def _gradient(
        self, choi: NDArray[np.complex128], n_vec: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """
        Finds gradient between expected and measured expectation values.
        """
        return -_unvec(np.conj(self._a_matrix.T) @ (n_vec / self._p_vec(choi)))
