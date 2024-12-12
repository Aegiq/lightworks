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

from ...sdk.circuit.photonic_compiler import CompiledPhotonicCircuit
from ...sdk.state import State
from ..utils import BackendError
from .abc_backend import BackendABC

# ruff: noqa: ARG002, D102


class FockBackend(BackendABC):
    """
    Base class for all backends. An outline of all possible functions should
    be included here.
    """

    def probability_amplitude(
        self, unitary: np.ndarray, input_state: list, output_state: list
    ) -> complex:
        raise BackendError(
            "Current backend does not implement probability_amplitude method."
        )

    def probability(
        self, unitary: np.ndarray, input_state: list, output_state: list
    ) -> float:
        raise BackendError(
            "Current backend does not implement probability method."
        )

    def full_probability_distribution(
        self, circuit: CompiledPhotonicCircuit, input_state: State
    ) -> dict:
        raise BackendError(
            "Current backend does not implement full_probability_distribution "
            "method."
        )
