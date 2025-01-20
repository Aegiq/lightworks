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

"""
Contains a collection of different useful functions for operations on matrices.
"""

import numpy as np
from numpy.typing import NDArray

from ...__settings import settings


def check_unitary(
    U: NDArray[np.complex128],  # noqa: N803
    precision: float | None = None,
) -> bool:
    """
    A function to check if a provided matrix is unitary according to a
    certain level of precision. If finds the product of the matrix with its
    hermitian conjugate and then checks it is unitary.

    Args:

        U (np.array) : The NxN matrix which we want to check is unitary.

        precision (float, optional) : The precision which the unitary
            matrix is checked according to. If there are large float errors
            this may need to be reduced.

    Returns:

        bool : A boolean to indicate whether or not the matrix is unitary.

    Raises:

        ValueError : Raised in the event that the matrix is not square as it
            cannot be unitary.

    """
    if precision is None:
        precision = settings.unitary_precision
    if U.shape[0] != U.shape[1]:
        raise ValueError("Unitary matrix must be square.")
    # Find hermitian conjugate and then product
    hc = np.conj(np.transpose(U))
    # Validate close according to tolerance
    return np.allclose(
        hc @ U, np.identity(U.shape[0], dtype=complex), rtol=0, atol=precision
    )


def add_mode_to_unitary(
    unitary: NDArray[np.complex128], add_mode: int
) -> NDArray[np.complex128]:
    """
    Adds a new mode (through inclusion of an extra row/column) to the provided
    unitary at the selected location.

    Args:

        unitary (np.ndarray) : The unitary to add a mode to.

        add_mode (int) : The location at which a new mode should be added to
            the circuit.

    Returns:

        np.ndarray : The converted unitary matrix.

    """
    dim = unitary.shape[0] + 1
    new_u = np.identity(dim, dtype=complex)
    # Diagonals
    new_u[:add_mode, :add_mode] = unitary[:add_mode, :add_mode]
    new_u[add_mode + 1 :, add_mode + 1 :] = unitary[add_mode:, add_mode:]
    # Off-diagonals
    new_u[:add_mode, add_mode + 1 :] = unitary[:add_mode, add_mode:]
    new_u[add_mode + 1 :, :add_mode] = unitary[add_mode:, :add_mode]
    return new_u
