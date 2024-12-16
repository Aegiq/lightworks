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

from .. import qubit
from ..sdk.circuit import PhotonicCircuit
from ..sdk.state import State

PAULI_MAPPING: dict[str, NDArray[np.complex128]] = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

# Pre-calculated density matrices for different quantum states
RHO_MAPPING: dict[str, NDArray[np.complex128]] = {
    "X+": np.array([[1, 1], [1, 1]], dtype=np.complex128) / 2,
    "X-": np.array([[1, -1], [-1, 1]], dtype=np.complex128) / 2,
    "Y+": np.array([[1, -1j], [1j, 1]], dtype=np.complex128) / 2,
    "Y-": np.array([[1, 1j], [-1j, 1]], dtype=np.complex128) / 2,
    "Z+": np.array([[1, 0], [0, 0]], dtype=np.complex128),
    "Z-": np.array([[0, 0], [0, 1]], dtype=np.complex128),
}

# Details the actual input state and transformation required to achieve a target
# input
r_transform = qubit.H()
r_transform.add(qubit.S())
INPUT_MAPPING: dict[str, tuple[State, PhotonicCircuit]] = {
    "X+": (State([1, 0]), qubit.H()),
    "X-": (State([0, 1]), qubit.H()),
    "Y+": (State([1, 0]), r_transform),
    "Y-": (State([0, 1]), r_transform),
    "Z+": (State([1, 0]), qubit.I()),
    "Z-": (State([0, 1]), qubit.I()),
}

# Details transformations required for different measurement types
_y_measure = PhotonicCircuit(2)
_y_measure.add(qubit.S())
_y_measure.add(qubit.Z())
_y_measure.add(qubit.H())
MEASUREMENT_MAPPING: dict[str, PhotonicCircuit] = {
    "X": qubit.H(),
    "Y": _y_measure,
    "Z": qubit.I(),
    "I": qubit.I(),
}
