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

from ..utils import DecompositionUnsuccessful


def reck_decomposition(unitary: np.ndarray) -> tuple[dict[str, float], list]:
    """
    Performs the triangular decomposition procedure for a provided unitary
    matrix.

    Args:

        unitary (np.ndarray) : The unitary matrix on which the decomposition
            should be performed.

    Returns:

        phase_map (dict) : A dictionary which details the calculated theta
            and phi values for each of the unit cells in the
            interferometer..

        D_prime (np.ndarray) : The remaining diagonal matrix after the
            decomposition has been completed.

    """
    n_modes = unitary.shape[0]
    phase_map = {}

    for i in range(0, n_modes - 1, 1):
        for j in range(n_modes - 1 - i):
            # Determine location to null
            loc = n_modes - 1 - i
            # Get elements from unitary
            u_ij = unitary[loc, j]
            u_ij1 = unitary[loc, j + 1]
            # Check if already nulled
            if abs(u_ij) < 1e-20:
                theta, phi = np.pi, 0
            else:
                # Calculate theta and phi
                theta = 2 * np.arctan(abs(u_ij1) / abs(u_ij))
                phi = np.angle(u_ij) - np.angle(u_ij1)
            # Create transformation matrix
            tr_ij = bs_matrix(j, j + 1, theta, phi, n_modes)
            # Null element
            unitary = unitary @ np.conj(tr_ij.T)
            phase_map[f"bs_{j + 2 * i}_{j}"] = theta
            phase_map[f"ps_{j + 2 * i}_{j}"] = phi

    # Check matrix has indeed been nulled by code, otherwise raise error
    if not check_null(unitary):
        raise DecompositionUnsuccessful(
            "Unable to successfully perform unitary decomposition procedure."
        )

    end_phases = [np.angle(unitary[i, i]) for i in range(n_modes)]

    return phase_map, end_phases


def bs_matrix(
    mode1: int, mode2: int, theta: float, phi: float, n_modes: int
) -> np.ndarray:
    """
    Desc
    """
    mat = np.identity(n_modes, dtype=complex)
    gp = 1j * np.exp(1j * theta / 2)
    mat[mode1, mode1] = -np.exp(1j * phi) * np.sin(theta / 2) * gp
    mat[mode1, mode2] = np.cos(theta / 2) * gp
    mat[mode2, mode1] = np.exp(1j * phi) * np.cos(theta / 2) * gp
    mat[mode2, mode2] = np.sin(theta / 2) * gp
    return mat


def check_null(mat: np.ndarray, precision: float = 1e-10) -> np.ndarray:
    """
    A function to check if a provided matrix has been nulled correctly by
    the algorithm.

    Args:

        mat (np.array) : The resultant nulled matrix

        precision (float, optional) : The precision which the matrix is
            checked according to. If there are large float errors this may
            need to be reduced.

        print_mat (bool, optional) : Select whether to print a cleaned
            version of the nulled matrix.

        plot_abs (bool, optional) : Select whether to produce a heatmap
            plot of the output value of the matrix.

    Returns:

        unitary (bool) : A boolean to indicate whether or not the matrix is
            unitary.

    """
    # Loop over each value and ensure it is the expected number to some
    # level of precision
    nulled = True
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            # Off diagonals
            if i != j and (
                np.real(mat[i, j] > precision) or np.imag(mat[i, j]) > precision
            ):
                nulled = False
    # Return whether matrix has been nulled or not
    return nulled
