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

from ..sdk.utils import check_random_seed
from .dists import Constant, Distribution


class ErrorModel:
    """
    Placeholder class for configuring an error model which can be introduced
    within an interferometer.
    """

    def __init__(self) -> None:
        self.bs_reflectivity = Constant(0.5)
        self.loss = Constant(0)

        return

    @property
    def bs_reflectivity(self) -> Distribution:
        """Returns currently in use beam splitter value distribution."""
        return self._bs_reflectivity

    @bs_reflectivity.setter
    def bs_reflectivity(self, distribution: Distribution) -> None:
        if not isinstance(distribution, Distribution):
            raise TypeError("bs_reflectivity should be a distribution object.")
        self._bs_reflectivity = distribution

    @property
    def loss(self) -> Distribution:
        """Returns currently in use loss value distribution."""
        return self._loss

    @loss.setter
    def loss(self, distribution: Distribution) -> None:
        if not isinstance(distribution, Distribution):
            raise TypeError("loss should be a distribution object.")
        self._loss = distribution

    def get_bs_reflectivity(self) -> int | float:
        """
        Returns a value for beam splitter reflectivity, which depends on the
        configuration of the error model.
        """
        return self._bs_reflectivity.value()

    def get_loss(self) -> int | float:
        """
        Returns a value for loss which depends on the configuration of the error
        model.
        """
        return self._loss.value()

    def _set_random_seed(self, r_seed: int | None) -> None:
        """
        Set the random seed for the error_model to produce repeatable results.
        """
        seed = check_random_seed(r_seed)
        # Create a rng to modify the seed by, ensuring two distributions produce
        # different values
        rng = random.default_rng(seed)
        # Set random seed in each property if present
        for prop in [self._bs_reflectivity, self._loss]:
            if hasattr(prop, "set_random_seed") and callable(
                prop.set_random_seed
            ):
                if seed is not None:
                    mod_seed = rng.integers(0, 2**31)
                    seed = seed * mod_seed
                prop.set_random_seed(seed)
