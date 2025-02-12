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
from numpy.typing import NDArray

from lightworks.sdk.utils.exceptions import DecompositionUnsuccessful
from lightworks.sdk.utils.matrix import check_unitary


def reck_decomposition(
    unitary: NDArray[np.complex128],
) -> tuple[dict[str, float], list[float]]:
    """
    Performs the triangular decomposition procedure for a provided unitary
    matrix.

    Args:

        unitary (np.ndarray) : The unitary matrix on which the decomposition
            should be performed.

    Returns:

        dict : A dictionary which stores the calculated phase shifter settings
            for each element of the interferometer

        list : A list of the residual phases which needs to be applied at the
            output of the system.

    """
    # Check matrix is unitary before decomposition
    if not check_unitary(unitary):
        raise ValueError("Provided matrix determined not to be unitary.")
    n_modes = unitary.shape[0]
    # Dictionary to store calculated phases
    phase_map: dict[str, float] = {}
    # Loop over each element in matrix
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
            unitary @= np.conj(tr_ij.T)
            phase_map[f"bs_{j + 2 * i}_{j}"] = theta
            phase_map[f"ps_{j + 2 * i}_{j}"] = phi

    # Check matrix has indeed been nulled by code, otherwise raise error
    if not check_null(unitary):
        raise DecompositionUnsuccessful(
            "Unable to successfully perform unitary decomposition procedure."
        )
    # Extract remaining end phases from diagonals of unitary
    end_phases = [np.angle(unitary[i, i]) for i in range(n_modes)]

    return phase_map, end_phases


def bs_matrix(
    mode1: int, mode2: int, theta: float, phi: float, n_modes: int
) -> NDArray[np.complex128]:
    """
    Generates a n_modes X n_modes matrix which implements the beam
    transformation of the unit cell between two modes.
    """
    mat = np.identity(n_modes, dtype=complex)
    gp = 1j * np.exp(1j * theta / 2)
    mat[mode1, mode1] = -np.exp(1j * phi) * np.sin(theta / 2) * gp
    mat[mode1, mode2] = np.cos(theta / 2) * gp
    mat[mode2, mode1] = np.exp(1j * phi) * np.cos(theta / 2) * gp
    mat[mode2, mode2] = np.sin(theta / 2) * gp
    return mat


def check_null(mat: NDArray[np.complex128], precision: float = 1e-10) -> bool:
    """
    A function to check if a provided matrix has been nulled correctly by
    the algorithm.

    Args:

        mat (np.array) : The resultant nulled matrix.

        precision (float, optional) : The precision which the matrix is
            checked according to. If there are large float errors this may
            need to be reduced.

    Returns:

        bool : Indicates if the original unitary was successfully nulled.

    """
    # Loop over each value and ensure it is the expected number to some
    # level of precision
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            # Check off diagonals are nulled
            if i != j and (
                np.real(mat[i, j] > precision) or np.imag(mat[i, j]) > precision
            ):
                return False  # Return false if any elements aren't
    # Return true if above loop passes
    return True
