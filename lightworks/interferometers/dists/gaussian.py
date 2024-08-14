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
from numpy import random

from .distribution import Distribution
from .utils import is_number


class Gaussian(Distribution):
    """
    Returns random values according to a Gaussian distribution with defined
    center and standard deviation. It can also be constrained to be between a
    minimum and maximum value to prevent issues with assigning invalid
    quantities. When a value is outside of the set bounds the distribution will
    be resampled until this is no longer the case. Note: care should be taken
    with setting minimum and maximum values as setting these to be too strict
    could significantly increase the time taken to produce a valid value.

    Args:

        center (float) : The center (mean) of the Gaussian distribution.

        deviation (float) : The standard deviation of the distribution.

        min_value (float | None) : The minimum allowed value for the
            distribution. Defaults to None, which will assign the min value to
            be - infinity.

        max_value (float | None) : The maximum allowed value for the
            distribution. Defaults to None, which will assign the max value to
            be + infinity.

    """

    def __init__(
        self,
        center: float,
        deviation: float,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        # Assign min\max to +/i infinity if None
        if min_value is None:
            min_value = -np.inf
        if max_value is None:
            max_value = np.inf
        # Type check all values
        is_number([center, deviation, min_value, max_value])
        # Then check min and max values are valid
        if max_value < min_value:
            raise ValueError("Max value cannot be less than min value.")
        # Then assign to attributes
        self._center = center
        self._deviation = deviation
        self._min_value = min_value
        self._max_value = max_value
        self._rng = random.default_rng()

    def __str__(self) -> str:
        if self._min_value == -np.inf and self._max_value == np.inf:
            contents = f"\u03bc = {self._center}, \u03c3 = {self._deviation}"
        else:
            contents = (
                f"\u03bc = {self._center}, \u03c3 = {self._deviation}, "
                f"bounds = [{self._min_value}, {self._max_value}]"
            )
        return f"Gaussian({contents})"

    def __repr__(self) -> str:
        return "lightworks.interferometers.dists." + str(self)

    def value(self) -> float:
        """Returns random value from the Gaussian distribution."""
        val = self._rng.normal(self._center, self._deviation)
        # Recalculate value until valid.
        while val < self._min_value or val > self._max_value:
            val = self._rng.normal(self._center, self._deviation)
        # Then return
        return val

    def set_random_seed(self, seed: int | None) -> None:
        """Used for setting the random seed for the model."""
        self._rng = random.default_rng(seed)
