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
    Base class for all distributions. Enforce that the class has a value method.
    """

    @abstractmethod
    def value(self) -> int | float:
        """Returns a value from the distribution on request."""
        pass


class Constant(Distribution):
    """
    Desc
    """

    def __init__(self, value: int | float) -> None:
        is_number(value)
        self._value = value

    def value(self) -> int | float:
        """Returns set constant value."""
        return self._value


class Gaussian(Distribution):
    """
    Desc
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
        """Returns value from the Gaussian distribution."""
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
    Desc
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
        """Returns value from range."""
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
