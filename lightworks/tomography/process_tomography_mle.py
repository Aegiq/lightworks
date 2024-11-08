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

from ..sdk.state import State
from .mappings import PAULI_MAPPING, RHO_MAPPING
from .process_tomography import ProcessTomography
from .utils import _get_tomo_measurements, unvec, vec

TOMO_INPUTS = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]


class MLEProcessTomography(ProcessTomography):
    """
    Runs quantum process tomography using the maximum likelihood estimation
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
        Desc
        """
        results = self._run_required_experiments()
        nij = {}
        for (in_state, meas), result in results.items():
            total = 0
            n_counts = 0
            for s, c in result.items():
                n_counts += c
                # Adjust multiplier to account for variation in eigenvalues
                multiplier = 1
                for j, gate in enumerate(meas.split(",")):
                    if gate == "I" or s[2 * j : 2 * j + 2] == State([1, 0]):
                        multiplier *= 1
                    elif s[2 * j : 2 * j + 2] == State([0, 1]):
                        multiplier *= -1
                total += multiplier * c

            nij[in_state, meas] = total / n_counts

        n_vec = []
        for n in nij.values():
            n_vec.append((1 + n) / 2)
            n_vec.append((1 - n) / 2)
        n_vec_a = np.array(n_vec) / len(nij)

        mle = MLETomographyAlgorithm(self.n_qubits)
        self._choi = mle.pgdb(n_vec_a)
        return self.choi

    def _run_required_experiments(self) -> dict:
        all_inputs = self._generate_all_inputs()
        all_measurements = self._generate_all_measurements()
        # Determine required input states and circuits
        all_circuits = []
        all_input_states = []
        for in_state in all_inputs:
            for meas in all_measurements:
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
        num_per_in = len(all_measurements)
        return {
            (in_state, meas): results[num_per_in * i + j]
            for i, in_state in enumerate(all_inputs)
            for j, meas in enumerate(all_measurements)
        }

    def _generate_all_inputs(self) -> list:
        all_inputs = list(TOMO_INPUTS)
        for _ in range(self.n_qubits - 1):
            all_inputs = [
                i1 + "," + i2 for i1 in all_inputs for i2 in TOMO_INPUTS
            ]
        return all_inputs

    def _generate_all_measurements(self) -> list:
        all_measurements = _get_tomo_measurements(self.n_qubits)
        # Remove all identity measurement as this is trivial
        all_measurements.pop(
            all_measurements.index(",".join("I" * self.n_qubits))
        )
        return all_measurements


class MLETomographyAlgorithm:
    """
    Desc
    """

    def __init__(self, n_qubits: int, stop_threshold: float = 1e-10) -> None:
        self.n_qubits = n_qubits
        self.stop_threshold = stop_threshold

        self._all_rhos = dict(RHO_MAPPING)
        self._all_pauli = dict(PAULI_MAPPING)
        self._input_basis = list(TOMO_INPUTS)
        for _ in range(n_qubits - 1):
            self._all_rhos = {
                s1 + s2: np.kron(p1, p2)
                for s1, p1 in self._all_rhos.items()
                for s2, p2 in RHO_MAPPING.items()
            }
            self._all_pauli = {
                s1 + "," + s2: np.kron(p1, p2)
                for s1, p1 in self._all_pauli.items()
                for s2, p2 in PAULI_MAPPING.items()
            }
            self._input_basis = [
                i + j for i in self._input_basis for j in TOMO_INPUTS
            ]
        self._meas_basis = _get_tomo_measurements(n_qubits)
        self._meas_basis.pop(
            self._meas_basis.index(",".join("I" * self.n_qubits))
        )

        self._a_matrix = self._a_mat()

    def pgdb(self, n_vec: np.ndarray, max_iter: int = 2000) -> np.ndarray:
        """
        Desc
        """
        dim = 2**self.n_qubits
        choi = np.identity(dim**2, dtype=complex) / dim

        current_cost = self._cost(choi, n_vec)

        mu = 3 / (2 * dim**2)
        gamma = 0.3

        for _ in range(max_iter):
            mod = (
                self._cptp_proj(
                    choi - 1 / mu * self._gradient(choi, n_vec),
                    max_iter=max_iter,
                )
                - choi
            )
            alpha = 1.0
            new_cost = self._cost(choi + alpha * mod, n_vec)
            thresh_value = gamma * np.trace(
                mod @ np.conj(self._gradient(choi.T, n_vec))
            )
            while new_cost > current_cost + alpha * thresh_value:
                alpha *= 0.5
                new_cost = self._cost(choi + alpha * mod, n_vec)

            choi += alpha * mod

            if current_cost - new_cost < 1e-12:
                break

            current_cost = new_cost

        else:
            warnings.warn("Max iterations exceeded.", stacklevel=1)

        return choi

    def _a_mat(self) -> np.ndarray:
        """
        Desc
        """
        dim1 = len(self._input_basis) * len(self._meas_basis) * 2
        dim2 = 2 ** (4 * self.n_qubits)
        a_mat = np.zeros((dim1, dim2), dtype=complex)
        id_mat = np.identity(2**self.n_qubits, dtype=complex)
        for i, in_s in enumerate(self._input_basis):
            for j, meas in enumerate(self._meas_basis):
                obs = self._all_pauli[meas]
                a_mat[2 * (len(self._meas_basis) * i + j), :] = vec(
                    np.kron(self._all_rhos[in_s], ((id_mat + obs) / 2).T)
                )[:]
                a_mat[2 * (len(self._meas_basis) * i + j) + 1, :] = vec(
                    np.kron(self._all_rhos[in_s], ((id_mat - obs) / 2).T)
                )[:]
        return a_mat / (2 ** (2 * self.n_qubits))

    def _p_vec(self, choi: np.ndarray) -> np.ndarray:
        """
        Desc
        """
        p = self._a_matrix @ vec(choi.T)
        return p.clip(1e-6)

    def _cost(self, choi: np.ndarray, n_vec: np.ndarray) -> np.ndarray:
        """
        Desc
        """
        return -np.real(n_vec.T @ np.log(self._p_vec(choi)))

    def _gradient(self, choi: np.ndarray, n_vec: np.ndarray) -> np.ndarray:
        """
        Desc
        """
        return -unvec(np.conj(self._a_matrix.T) @ (n_vec / self._p_vec(choi)))

    def _cptp_proj(self, choi: np.ndarray, max_iter: int = 1000) -> np.ndarray:
        """
        Desc
        """
        x_0 = vec(choi)
        dim = x_0.shape[0]
        p_0 = np.array([0] * dim)
        q_0 = np.array([0] * dim)
        y_0 = np.array([0] * dim)

        for _ in range(max_iter):
            y_k = vec(self._tp_proj(unvec(x_0 + p_0)))
            p_k = x_0 + p_0 - y_k
            x_k = vec(self._cp_proj(unvec(y_k + q_0)))
            q_k = y_k + q_0 - x_k

            if (
                np.linalg.norm(p_0 - p_k) ** 2
                + np.linalg.norm(q_0 - q_k) ** 2
                + abs(2 * np.conj(p_0) @ (x_k - x_0))
                + abs(2 * np.conj(q_0) @ (y_k - y_0))
            ) < 10**-4:
                break

            y_0, p_0, x_0, q_0 = y_k, p_k, x_k, q_k

        return unvec(x_k)

    def _cp_proj(self, choi: np.ndarray) -> np.ndarray:
        """
        Desc
        """
        vals, vecs = np.linalg.eigh(choi)
        d = np.diag([max(v, 0) for v in vals])
        return vecs @ d @ np.conj(vecs.T)

    def _tp_proj(self, choi: np.ndarray) -> np.ndarray:
        """
        Desc
        """
        dim = int(choi.shape[0] ** 0.5)

        partial_trace = choi.reshape(np.tile([dim, dim], 2))
        partial_trace = np.einsum(partial_trace, [0, 1, 2, 1])
        partial_trace = partial_trace.reshape(dim, dim)

        diff = partial_trace - np.identity(dim)
        return choi - np.kron(diff / dim, np.identity(dim))
