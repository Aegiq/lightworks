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

from types import NoneType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import unitary_group


def process_random_seed(seed: Any) -> int | None:
    """
    Takes a provided random seed, validates it is an integer or None, and if
    not will try to convert this. If this isn't possible then an exception will
    be raised.
    """
    if not isinstance(seed, int | NoneType):
        # Try to convert seed to an integer
        try:
            int(seed)
        except Exception as e:
            raise TypeError("Random seed must be an integer.") from e
        # If can convert then check equivalence
        if int(seed) == seed:
            seed = int(seed)
        else:
            raise TypeError("Random seed must be an integer.")
    elif isinstance(seed, bool):
        raise TypeError("Random seed must be an integer.")
    return seed


def random_unitary(
    N: int,  # noqa: N803
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    Generate a random NxN unitary matrix. Seed can be used to produce the same
    unitary each time the function is called.

    Args:

        N (int) : The dimension of the random unitary that is to be generated.

        seed (int | None, optional) : Specify a random seed to repeatedly
            produce the same unitary matrix on each function call. Defaults to
            None, which will produce a random matrix on each call.

    Returns:

        np.ndarray : The created random unitary matrix.

    Raises:

        TypeError : In cases where the supplied random seed is not an integer
            value.

    """
    seed = process_random_seed(seed)
    return unitary_group.rvs(N, random_state=seed)


def random_permutation(
    N: int,  # noqa: N803
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    Generate a random NxN permutation. Seed can be used to produce the same
    unitary each time the function is called.

    Args:

        N (int) : The dimension of the random permutation that is to be
            generated.

        seed (int | None, optional) : Specify a random seed to repeatedly
            produce the same unitary matrix on each function call. Defaults to
            None, which will produce a random matrix on each call.

    Returns:

        np.ndarray : The created random permutation matrix.

    Raises:

        TypeError : In cases where the supplied random seed is not an integer
            value.

    """
    seed = process_random_seed(seed)
    rng = np.random.default_rng(seed)
    return rng.permutation(np.identity(N, dtype=complex))
