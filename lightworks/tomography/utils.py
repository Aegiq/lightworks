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
from numpy.typing import NDArray
from scipy.linalg import sqrtm

from lightworks.sdk.state import State

from .mappings import MEASUREMENT_MAPPING, PAULI_MAPPING

# NOTE: This file is auto-documented, so any functions not intended to be public
# should begin with _


def state_fidelity(
    rho: NDArray[np.complex128], rho_exp: NDArray[np.complex128]
) -> float:
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


def process_fidelity(
    choi: NDArray[np.complex128], choi_exp: NDArray[np.complex128]
) -> float:
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


def density_from_state(
    state: list[complex] | NDArray[np.complex128],
) -> NDArray[np.complex128]:
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


def choi_from_unitary(
    unitary: NDArray[np.complex128],
) -> NDArray[np.complex128]:
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


def _vec(mat: NDArray[Any]) -> NDArray[Any]:
    """
    Applies flatten operation to a provided matrix to convert it into a vector.
    """
    return mat.flatten()


def _unvec(mat: NDArray[Any]) -> NDArray[Any]:
    """
    Takes a provided vector and converts it into a square matrix.
    """
    dim = int(mat.shape[0] ** 0.5)
    return mat.reshape(dim, dim)


@multimethod
def _combine_all(value: Any, n: int) -> None:  # noqa: ARG001
    """
    Combines all elements of provided value with itself n number of times.
    """
    raise TypeError("combine_all method not implemented for provided type.")


@_combine_all.register
def _combine_all_list(value: list[str], n: int) -> list[str]:
    """
    Sums string values within list.
    """
    result = list(value)
    for _ in range(n - 1):
        result = [v1 + "," + v2 for v1 in result for v2 in value]
    return result


@_combine_all.register
def _combine_all_list_array(
    value: list[NDArray[Any]], n: int
) -> list[NDArray[Any]]:
    """
    Performs tensor product of all combinations of arrays within list.
    """
    result = list(value)
    for _ in range(n - 1):
        result = [np.kron(v1, v2) for v1 in result for v2 in value]
    return result


@_combine_all.register
def _combine_all_dict_mat(
    value: dict[str, NDArray[Any]], n: int
) -> dict[str, NDArray[Any]]:
    """
    Sums keys of dictionary and performs tensor products of the dictionary
    values.
    """
    result = dict(value)
    for _ in range(n - 1):
        result = {
            k1 + "," + k2: np.kron(v1, v2)
            for k1, v1 in result.items()
            for k2, v2 in value.items()
        }
    return result


def _get_tomo_measurements(
    n_qubits: int, remove_trivial: bool = False
) -> list[str]:
    """
    Returns all measurements required for a state tomography of n qubits.

    Args:

        n_qubits (int) : The number of qubits used in the tomography.

        remove_trivial (bool) : Allows for removal of the trivial I*n_qubits
            measurement when this is not required

    Returns:

        list : A list of the measurement combinations for tomography.

    """
    all_meas = _combine_all(list(MEASUREMENT_MAPPING.keys()), n_qubits)
    if remove_trivial:
        all_meas.pop(all_meas.index(",".join("I" * n_qubits)))
    return all_meas


def _get_required_tomo_measurements(
    n_qubits: int, remove_trivial: bool = False
) -> tuple[list[str], dict[str, str]]:
    """
    Calculates reduced list of required measurements assuming that any
    measurements in the I basis can be replaced with a Z measurement.
    A dictionary which maps the full measurements to the reduced basis is
    also returned.

    Args:

        n_qubits (int) : The number of qubits used in the tomography.

        remove_trivial (bool) : Allows for removal of the trivial I*n_qubits
            measurement when this is not required

    Returns:

        list : A list of the minimum required measurement combinations for
            tomography.

        dict : A mapping between the full set of measurement operators and
            the required minimum set.

    """
    mapping = {
        c: c.replace("I", "Z")
        for c in _get_tomo_measurements(n_qubits, remove_trivial=remove_trivial)
    }
    req_measurements = list(set(mapping.values()))
    return req_measurements, mapping


def _calculate_expectation_value(
    measurement: str, results: dict[State, int]
) -> float:
    """
    Calculates the expectation value for a given measurement and set of
    results.

    Args:

        measurement (str) : The measurement operator used for the
            computation.

        results (dict) : A dictionary of measured output states and counts.

    Returns:

        float : The calculated expectation value.

    """
    expectation = 0
    n_counts = 0
    for state, counts in results.items():
        n_counts += counts
        # Adjust multiplier to account for variation in eigenvalues
        multiplier = 1
        for j, gate in enumerate(measurement.split(",")):
            if gate == "I" or state[2 * j : 2 * j + 2] == State([1, 0]):
                multiplier *= 1
            elif state[2 * j : 2 * j + 2] == State([0, 1]):
                multiplier *= -1
            else:
                msg = (
                    f"An invalid state {state[2 * j : 2 * j + 2]} was found"
                    " in the results. This does not correspond to a valid "
                    "value for dual-rail encoded qubits."
                )
                raise ValueError(msg)
        expectation += multiplier * counts
    return expectation / n_counts


def _calculate_density_matrix(
    results: dict[str, dict[State, int]], n_qubits: int
) -> NDArray[np.complex128]:
    """
    Calculates the density matrix using a provided dictionary of results
    data.

    Args:

        results (dict) : Contains data of outputs and counts for each
            corresponding set of measurement indices.

        n_qubits (int) : The number of qubits in the experiment.

    Returns:

        np.ndarray : The calculated density matrix.


    """
    # Process results to find density matrix
    rho = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for measurement, result in results.items():
        expectation = _calculate_expectation_value(measurement, result)
        expectation /= 2**n_qubits
        # Calculate tensor product of the operators used
        ops = measurement.split(",")
        mat = PAULI_MAPPING[ops[0]]
        for g in ops[1:]:
            mat = np.kron(mat, PAULI_MAPPING[g])
        # Updated density matrix
        rho += expectation * mat
    return rho
