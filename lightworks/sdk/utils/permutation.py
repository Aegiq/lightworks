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


def permutation_mat_from_swaps_dict(
    swaps: dict[int, int], n_modes: int
) -> NDArray[np.complex128]:
    """
    Calculates the permutation unitary for a given dictionary of swaps across
    the n_modes of a circuit.

    Args:

        swaps (dict) : The dictionary containing the target mode swaps.

        n_modes (int) : The number of modes in the circuit. If this is not the
            number of modes then an incorrect dimension unitary will be
            returned.

    Returns:

        np.ndarray : The determined permutation matrix for the provided set of
            mode swaps.

    """
    if not isinstance(swaps, dict):
        raise TypeError("swaps should be a dictionary object.")

    # Add in missing modes from swap dictionary
    full_swaps = {}
    for m in range(n_modes):
        full_swaps[m] = swaps.get(m, m)
    # Create swap unitary
    permutation = np.zeros((n_modes, n_modes), dtype=complex)
    for i, j in full_swaps.items():
        permutation[j, i] = 1

    return permutation
