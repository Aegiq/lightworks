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

import warnings

import numpy as np
from numpy.typing import NDArray

from .mappings import PAULI_MAPPING, RHO_MAPPING
from .process_tomography import ProcessTomography
from .utils import (
    _calculate_expectation_value,
    _combine_all,
    _get_tomo_measurements,
    _unvec,
    _vec,
    process_fidelity,
)

TOMO_INPUTS = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]


class MLEProcessTomography(ProcessTomography):
    """
    Runs quantum process tomography using the maximum likelihood estimation
    method.

    Args:

        n_qubits (int) : The number of qubits that will be used as part of the
            tomography.

        base_circuit (PhotonicCircuit) : An initial circuit which produces the
            required output state and can be modified for performing tomography.
            It is required that the number of circuit input modes equals 2 * the
            number of qubits.

        experiment (Callable) : A function for performing the required
            tomography experiments. This should accept a list of circuits and a
            list of inputs and then return a list of results to process.

        experiment_args (list | None) : Optionally provide additional arguments
            which will be passed directly to the experiment function.

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

    def process(self) -> NDArray[np.complex128]:
        """
        Performs process tomography with the configured elements and calculates
        the choi matrix using maximum likelihood estimation.

        Returns:

            np.ndarray : The calculated choi matrix for the process.

        """
        all_inputs = _combine_all(TOMO_INPUTS, self.n_qubits)
        results = self._run_required_experiments(all_inputs)
        nij = {}
        # Convert counts to an expectation value
        for (in_state, meas), result in results.items():
            # Remove trivial measurement here
            if meas == ",".join("I" * self.n_qubits):
                continue
            nij[in_state, meas] = _calculate_expectation_value(meas, result)
        # Run MLE algorithm and get choi
        mle = MLETomographyAlgorithm(self.n_qubits)
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

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits

        # Pre-calculate all required n_qubit pauli & density matrices + the
        # inputs and measurement basis for the tomography
        self._all_rhos = _combine_all(RHO_MAPPING, self.n_qubits)
        self._all_pauli = _combine_all(PAULI_MAPPING, self.n_qubits)
        self._input_basis = _combine_all(TOMO_INPUTS, self.n_qubits)
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
                that the algorithm can do, defaults to 1000.

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
            mod = self._cptp_proj(
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

    def _cptp_proj(
        self, choi: NDArray[np.complex128], max_iter: int = 1000
    ) -> NDArray[np.complex128]:
        """
        Performs the CPTP algorithm, ensuring the matrix is completely positive
        and trace preserving. This uses a series of repeat TP and CP steps until
        the choi matrix converges.
        """
        x_0 = _vec(choi)
        dim = x_0.shape[0]
        # Initialize most quantities as a zero matrix
        p_0 = np.array([0] * dim)
        q_0 = np.array([0] * dim)
        y_0 = np.array([0] * dim)
        # Run for maximum number of iterations
        for _ in range(max_iter):
            # Calculate updated quantiles
            y_k = _vec(self._tp_proj(_unvec(x_0 + p_0)))
            p_k = x_0 + p_0 - y_k
            x_k = _vec(self._cp_proj(_unvec(y_k + q_0)))
            q_k = y_k + q_0 - x_k
            # Stopping condition (see paper)
            if (
                np.linalg.norm(p_0 - p_k) ** 2
                + np.linalg.norm(q_0 - q_k) ** 2
                + abs(2 * np.conj(p_0) @ (x_k - x_0))
                + abs(2 * np.conj(q_0) @ (y_k - y_0))
            ) < 10**-4:
                break
            # Update values
            y_0, p_0, x_0, q_0 = y_k, p_k, x_k, q_k

        return _unvec(x_k)

    def _cp_proj(self, choi: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Performs the CP part of the projection by enforcing that the eigenvalues
        of the matrix are all positive.
        """
        vals, vecs = np.linalg.eigh(choi)
        d = np.diag([max(v, 0) for v in vals])
        return vecs @ d @ np.conj(vecs.T)

    def _tp_proj(self, choi: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Performs the TP part of the projection by enforcing that the partial
        trace of the choi matrix is equal to the identity matrix.
        """
        # Compute partial trace
        dim = int(choi.shape[0] ** 0.5)
        partial_trace = choi.reshape(np.tile([dim, dim], 2))
        partial_trace = np.einsum(partial_trace, np.array([0, 1, 2, 1]))  # type: ignore[arg-type]
        partial_trace = partial_trace.reshape(dim, dim)
        # Then find the variation and subtract this from the choi matrix
        variation = partial_trace - np.identity(dim)
        return choi - np.kron(variation / dim, np.identity(dim))
