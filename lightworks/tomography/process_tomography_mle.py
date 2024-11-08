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
from types import FunctionType
from typing import Callable

import numpy as np

from .. import qubit
from ..sdk.circuit import Circuit
from ..sdk.state import State
from .utils import (
    MEASUREMENT_MAPPING,
    PAULI_MAPPING,
    process_fidelity,
    unvec,
    vec,
)

TOMO_INPUTS = ["X0", "X1", "Y0", "Y1", "Z0", "Z1"]
TOMO_MEASUREMENTS = ["X", "Y", "Z", "I"]

r_transform = qubit.H()
r_transform.add(qubit.S())
INPUT_MAPPING: dict[str, tuple[State, Circuit]] = {
    "X0": (State([1, 0]), qubit.H()),
    "X1": (State([0, 1]), qubit.H()),
    "Y0": (State([1, 0]), r_transform),
    "Y1": (State([0, 1]), r_transform),
    "Z0": (State([1, 0]), qubit.I()),
    "Z1": (State([0, 1]), qubit.I()),
}
RHO_MAPPING: dict[str, np.ndarray] = {
    "X0": np.array([[1, 1], [1, 1]]) / 2,
    "X1": np.array([[1, -1], [-1, 1]]) / 2,
    "Y0": np.array([[1, -1j], [1j, 1]]) / 2,
    "Y1": np.array([[1, 1j], [-1j, 1]]) / 2,
    "Z0": np.array([[1, 0], [0, 0]]),
    "Z1": np.array([[0, 0], [0, 1]]),
    "I0": np.array([[1, 0], [0, 1]]),
    "I1": np.array([[1, 0], [0, 1]]),
}


class ProcessTomographyMLE:
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
        # Type check inputs
        if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
            raise TypeError("Number of qubits should be an integer.")
        if not isinstance(base_circuit, Circuit):
            raise TypeError("Base circuit should be a circuit object.")

        if 2 * n_qubits != base_circuit.input_modes:
            msg = (
                "Number of circuit input modes does not match the amount "
                "required for the specified number of qubits, expected "
                f"{2 * n_qubits}."
            )
            raise ValueError(msg)

        self._n_qubits = n_qubits
        self._base_circuit = base_circuit
        self.experiment = experiment
        self.experiment_args = experiment_args
        self._choi: np.ndarray

    @property
    def base_circuit(self) -> Circuit:
        """
        The base circuit which is to be modified as part of the tomography
        calculations.
        """
        return self._base_circuit

    @property
    def n_qubits(self) -> int:
        """
        The number of qubits within the system.
        """
        return self._n_qubits

    @property
    def experiment(self) -> Callable:
        """
        A function to call which runs the required experiments. This should
        accept a list of circuits as a single argument and then return a list
        of the corresponding results, with each result being a dictionary or
        Results object containing output states and counts.
        """
        return self._experiment

    @experiment.setter
    def experiment(self, value: Callable) -> None:
        if not isinstance(value, FunctionType):
            raise TypeError(
                "Provided experiment should be a function which accepts a list "
                "of circuits and returns a list of results containing only the "
                "qubit modes."
            )
        self._experiment = value

    @property
    def choi(self) -> np.ndarray:
        """Returns the calculate choi matrix for a circuit."""
        if not hasattr(self, "_choi"):
            raise AttributeError(
                "Choi matrix has not yet been calculated, this can be achieved "
                "with the process method."
            )
        return self._choi

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

    def fidelity(self, choi_exp: np.ndarray) -> float:
        """
        Calculates fidelity of the calculated choi matrix compared to the
        expected one.
        """
        return process_fidelity(self.choi, choi_exp)

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
            circ.add(MEASUREMENT_MAPPING[op], 2 * i)
        return circ, in_state

    def _generate_all_inputs(self) -> list:
        all_inputs = list(TOMO_INPUTS)
        for _ in range(self.n_qubits - 1):
            all_inputs = [
                i1 + "," + i2 for i1 in all_inputs for i2 in TOMO_INPUTS
            ]
        return all_inputs

    def _generate_all_measurements(self) -> list:
        all_measurements = list(TOMO_MEASUREMENTS)
        for _ in range(self.n_qubits - 1):
            all_measurements = [
                i1 + "," + i2
                for i1 in all_measurements
                for i2 in TOMO_MEASUREMENTS
            ]
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
        self._meas_basis = list(TOMO_MEASUREMENTS)
        for _ in range(n_qubits - 1):
            self._all_rhos = {
                s1 + s2: np.kron(p1, p2)
                for s1, p1 in self._all_rhos.items()
                for s2, p2 in RHO_MAPPING.items()
            }
            self._all_pauli = {
                s1 + s2: np.kron(p1, p2)
                for s1, p1 in self._all_pauli.items()
                for s2, p2 in PAULI_MAPPING.items()
            }
            self._input_basis = [
                i + j for i in self._input_basis for j in TOMO_INPUTS
            ]
            self._meas_basis = [
                i + j for i in self._meas_basis for j in TOMO_MEASUREMENTS
            ]
        self._meas_basis.pop(self._meas_basis.index("I" * n_qubits))

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
