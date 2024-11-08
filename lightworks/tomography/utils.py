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

from typing import Any

import numpy as np
from multimethod import multimethod
from scipy.linalg import sqrtm

from .mappings import MEASUREMENT_MAPPING


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


@multimethod
def combine_all(value: Any, n: int) -> None:  # noqa: ARG001
    """
    Combines all elements of provided value with itself n number of times.
    """
    raise TypeError("combine_all method not implemented for provided type.")


@combine_all.register
def _combine_all_list(value: list, n: int) -> list:
    result = list(value)
    for _ in range(n - 1):
        result = [v1 + v1 for v1 in result for v2 in value]
    return result


@combine_all.register
def _combine_all_dict_mat(value: dict[Any, np.ndarray], n: int) -> dict:
    result = dict(value)
    for _ in range(n - 1):
        result = {
            k1 + k2: v1 + v2
            for k1, v1 in result.items()
            for k2, v2 in value.items()
        }
    return result


def _get_tomo_measurements(n_qubits: int) -> list[str]:
    """
    Returns all measurements required for a state tomography of n qubits.

    Args:

        n_qubits (int) : The number of qubits used in the tomography.

    Returns:

        list : A list of the measurement combinations for tomography.

    """
    # Find all measurement combinations
    measurements = list(MEASUREMENT_MAPPING.keys())
    for _i in range(n_qubits - 1):
        measurements = [
            g1 + "," + g2 for g1 in measurements for g2 in MEASUREMENT_MAPPING
        ]
    return measurements


def _get_required_tomo_measurements(n_qubits: int) -> tuple[list, dict]:
    """
    Calculates reduced list of required measurements assuming that any
    measurements in the I basis can be replaced with a Z measurement.
    A dictionary which maps the full measurements to the reduced basis is
    also returned.

    Args:

        n_qubits (int) : The number of qubits used in the tomography.

    Returns:

        list : A list of the minimum required measurement combinations for
            tomography.

        dict : A mapping between the full set of measurement operators and
            the required minimum set.

    """
    mapping = {c: c.replace("I", "Z") for c in _get_tomo_measurements(n_qubits)}
    req_measurements = list(set(mapping.values()))
    return req_measurements, mapping
