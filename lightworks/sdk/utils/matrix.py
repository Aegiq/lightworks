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

from lightworks.__settings import settings


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
