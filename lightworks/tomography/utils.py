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
from scipy.linalg import sqrtm

from .. import qubit
from ..sdk.circuit import Circuit
from ..sdk.state import State

PAULI_MAPPING: dict[str, np.ndarray] = {
    "I": np.array([[1, 0], [0, 1]]),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}

RHO_MAPPING: dict[str, np.ndarray] = {
    "X+": np.array([[1, 1], [1, 1]]) / 2,
    "X-": np.array([[1, -1], [-1, 1]]) / 2,
    "Y+": np.array([[1, -1j], [1j, 1]]) / 2,
    "Y-": np.array([[1, 1j], [-1j, 1]]) / 2,
    "Z+": np.array([[1, 0], [0, 0]]),
    "Z-": np.array([[0, 0], [0, 1]]),
}

r_transform = qubit.H()
r_transform.add(qubit.S())
INPUT_MAPPING: dict[str, tuple[State, Circuit]] = {
    "X+": (State([1, 0]), qubit.H()),
    "X-": (State([0, 1]), qubit.H()),
    "Y+": (State([1, 0]), r_transform),
    "Y-": (State([0, 1]), r_transform),
    "Z+": (State([1, 0]), qubit.I()),
    "Z-": (State([0, 1]), qubit.I()),
}

_y_measure = Circuit(2)
_y_measure.add(qubit.S())
_y_measure.add(qubit.Z())
_y_measure.add(qubit.H())
MEASUREMENT_MAPPING = {
    "X": qubit.H(),
    "Y": _y_measure,
    "Z": qubit.I(),
    "I": qubit.I(),
}


def state_fidelity(rho: np.ndarray, rho_exp: np.ndarray) -> float:
    """
    Calculates the fidelity of the density matrix for a quantum state against
    the expected density matrix.

    Args:

        rho (np.ndarray) : The calculated density matrix of the quantum state.

        rho_exp (np.ndarray) : The expected density matrix.

    Returns:

        float : The calculated fidelity value.

    """
    rho_exp = np.array(rho_exp)
    rho_root = sqrtm(np.array(rho))
    if rho_root.shape != rho_exp.shape:
        msg = (
            "Mismatch in dimensions between provided density matrices, "
            f"{rho_root.shape} & {rho_exp.shape}."
        )
        raise ValueError(msg)
    inner = rho_root @ rho_exp @ rho_root
    return abs(np.trace(sqrtm(inner)))


def process_fidelity(choi: np.ndarray, choi_exp: np.ndarray) -> float:
    """
    Calculates the fidelity of a process compared to an expected choi matrix.

    Args:

        choi (np.ndarray) : The calculated choi matrix for the process.

        choi_exp (np.ndarray) : The expected choi matrix.

    Returns:

        float : The calculated fidelity value.

    """
    if choi.shape != choi_exp.shape:
        msg = (
            "Mismatch in dimensions between provided density matrices, "
            f"{choi.shape} & {choi_exp.shape}."
        )
        raise ValueError(msg)
    n_qubits = int(np.emath.logn(4, choi.shape[0]))
    return state_fidelity(choi / 2**n_qubits, choi_exp / 2**n_qubits)


def density_from_state(state: list | np.ndarray) -> np.ndarray:
    """
    Calculates the expected density matrix from a given state.

    Args:

        state (list | np.ndarray) : The vector representation of the state for
            which the density matrix should be calculated.

    Returns:

        np.ndarray : The calculated density matrix.

    """
    state = np.array(state)
    return np.outer(state, np.conj(state.T))


def choi_from_unitary(unitary: np.ndarray) -> np.ndarray:
    """
    Calculates the expected choi matrix from a given unitary representation of a
    process.

    Args:

        unitary (np.ndarray) : The unitary representation of the gate.

    Returns:

        np.ndarray : The calculated choi matrix.

    """
    unitary = np.array(unitary)
    return np.outer(unitary.flatten(), np.conj(unitary.flatten()))


def vec(mat: np.ndarray) -> np.ndarray:
    """
    Applies flatten operation to a provided matrix to convert it into a vector.
    """
    return mat.flatten()


def unvec(mat: np.ndarray) -> np.ndarray:
    """
    Takes a provided vector and converts it into a square matrix.
    """
    dim = int(mat.shape[0] ** 0.5)
    return mat.reshape(dim, dim)
