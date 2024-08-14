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

from ..sdk.utils import process_random_seed
from .dists import Constant, Distribution


class ErrorModel:
    """
    Allows for configuration of an error model which can be applied to different
    components of the interferometers.
    """

    def __init__(self) -> None:
        self.bs_reflectivity = Constant(0.5)
        self.loss = Constant(0)
        self.phase_offset = Constant(0)

        return

    def __str__(self) -> str:
        return (
            f"ErrorModel(bs_reflectivity = {self.bs_reflectivity}, "
            f"loss = {self.loss})"
        )

    def __repr__(self) -> str:
        return "lightworks.interferometers." + str(self)

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

    @property
    def phase_offset(self) -> Distribution:
        """Returns currently in use phase_offset value distribution."""
        return self._phase_offset

    @phase_offset.setter
    def phase_offset(self, distribution: Distribution) -> None:
        if not isinstance(distribution, Distribution):
            raise TypeError("phase_offset should be a distribution object.")
        self._phase_offset = distribution

    def get_bs_reflectivity(self) -> float:
        """
        Returns a value for beam splitter reflectivity, which depends on the
        configuration of the error model.
        """
        return self._bs_reflectivity.value()

    def get_loss(self) -> float:
        """
        Returns a value for loss, which depends on the configuration of the
        error model.
        """
        return self._loss.value()

    def get_phase_offset(self) -> float:
        """
        Returns a value for phase offset, which depends on the configuration of
        the error model.
        """
        return self._phase_offset.value()

    def _set_random_seed(self, r_seed: int | None) -> None:
        """
        Set the random seed for the error_model to produce repeatable results.
        """
        seed = process_random_seed(r_seed)
        # Create a rng to modify the seed by, ensuring two distributions produce
        # different values
        rng = random.default_rng(seed)
        # Set random seed in each property if present
        for prop in [self._bs_reflectivity, self._loss, self._phase_offset]:
            if hasattr(prop, "set_random_seed") and callable(
                prop.set_random_seed
            ):
                if seed is not None:
                    seed = rng.integers(2**31 - 1)
                prop.set_random_seed(seed)
