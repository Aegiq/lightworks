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

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
from numpy import random


class Distribution(metaclass=ABCMeta):
    """
    Base class for all distributions. Enforces that any created distributions
    have the required value method, which returns a singular value on request.
    If the distribution requires randomness, then the set_random_seed method
    should be added, which accepts a seed and sets this for the rng to allow
    for repeatability where required.
    """

    @abstractmethod
    def value(self) -> int | float:
        """Returns a value from the distribution on request."""
        pass


class Constant(Distribution):
    """
    Implements returns of a singular constant value for the assigned quantity.

    Args:

        value (int | float) : The constant value to use.

    """

    def __init__(self, value: int | float) -> None:
        is_number(value)
        self._value = value

    def value(self) -> int | float:
        """Returns set constant value."""
        return self._value


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

        center (int | float) : The center (mean) of the Gaussian distribution.

        deviation (int | float) : The standard deviation of the distribution.

        min_value (int | float | None) : The minimum allowed value for the
            distribution. Defaults to None, which will assign the min value to
            be - infinity.

        max_value (int | float | None) : The maximum allowed value for the
            distribution. Defaults to None, which will assign the max value to
            be + infinity.

    """

    def __init__(
        self,
        center: int | float,
        deviation: int | float,
        min_value: int | float | None = None,
        max_value: int | float | None = None,
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

    def value(self) -> int | float:
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


class TopHat(Distribution):
    """
    Returns random value according to a uniform distribution between two values.

    Args:

        min_value (int | float) : The minimum value of the distribution.

        max_value (int | float) : The maximum value of the distribution.

    """

    def __init__(self, min_value: int | float, max_value: int | float) -> None:
        # Type check all values
        is_number([min_value, max_value])
        # Then check min and max values are valid
        if max_value < min_value:
            raise ValueError("Max value cannot be less than min value.")
        # Then assign to attributes
        self._min_value = min_value
        self._max_value = max_value
        self._rng = random.default_rng()

    def value(self) -> int | float:
        """Returns random value from within set range."""
        return (
            self._min_value
            + (self._max_value - self._min_value) * self._rng.random()
        )

    def set_random_seed(self, seed: int | None) -> None:
        """Used for setting the random seed for the model."""
        self._rng = random.default_rng(seed)


def is_number(value: Any | list[Any]) -> None:
    """
    Function to check if the provided value is float or integer.
    """
    if not isinstance(value, (list, tuple)):
        value = [value]
    for v in value:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise TypeError("Distribution values should be a float or integer.")
