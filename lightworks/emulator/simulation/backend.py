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

from .permanent import Permanent
from .determinant import Determinant
from ..utils import BackendError
from ...sdk import State

from numpy import ndarray

class Backend:
    """General backend for handling both types of statistics."""
    @staticmethod
    def calculate(U: ndarray, in_state: State, out_state: State,
                  statistic_type: str) -> complex:
        if statistic_type == "bosonic":
            return Permanent.calculate(U, in_state, out_state)
        elif statistic_type == "fermionic":
            return Determinant.calculate(U, in_state, out_state)
        else:
            raise BackendError(
                "statistic_type should be 'bosonic' or 'fermionic'.")
    