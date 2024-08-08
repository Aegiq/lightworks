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

from numpy import random

from .distribution import Distribution
from .utils import is_number


class TopHat(Distribution):
    """
    Returns random value according to a uniform distribution between two values.

    Args:

        min_value (float) : The minimum value of the distribution.

        max_value (float) : The maximum value of the distribution.

    """

    def __init__(self, min_value: float, max_value: float) -> None:
        # Type check all values
        is_number([min_value, max_value])
        # Then check min and max values are valid
        if max_value < min_value:
            raise ValueError("Max value cannot be less than min value.")
        # Then assign to attributes
        self._min_value = min_value
        self._max_value = max_value
        self._rng = random.default_rng()

    def __str__(self) -> str:
        return f"TopHat({self._min_value}, {self._max_value})"

    def __repr__(self) -> str:
        return "lightworks.interferometers.dists." + str(self)

    def value(self) -> float:
        """Returns random value from within set range."""
        return (
            self._min_value
            + (self._max_value - self._min_value) * self._rng.random()
        )

    def set_random_seed(self, seed: int | None) -> None:
        """Used for setting the random seed for the model."""
        self._rng = random.default_rng(seed)
