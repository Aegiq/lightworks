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
Dedicated unitary component for implementing unitary matrices on a circuit.
"""

import numpy as np
from numpy.typing import NDArray

from .photonic_circuit import PhotonicCircuit
from .photonic_components import UnitaryMatrix


class Unitary(PhotonicCircuit):
    """
    Create a circuit which implements the target provided unitary across all of
    its modes.

    Args:

        unitary (np.ndarray) : The target NxN unitary matrix which is to be
            implemented.

    """

    def __init__(
        self, unitary: NDArray[np.complex128], label: str = "U"
    ) -> None:
        unitary = np.array(unitary)
        super().__init__(int(unitary.shape[0]))
        self._PhotonicCircuit__circuit_spec = [UnitaryMatrix(0, unitary, label)]

        return
