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

from ..sdk.utils import check_random_seed
from .dists import Constant, Distribution


class ErrorModel:
    """
    Placeholder class for configuring an error model which can be introduced
    within an interferometer.
    """

    def __init__(self) -> None:
        self.bs_reflectivity_dist = Constant(0.5)
        self.loss_dist = Constant(0)

        return

    @property
    def bs_reflectivity(self) -> int | float:
        """
        Returns a value for beam splitter reflectivity, which depends on the
        configuration of the error model.
        """
        return self._bs_reflectivity_dist.value()

    @property
    def bs_reflectivity_dist(self) -> int | float:
        """Returns currently in use beam splitter value distribution."""
        return self._bs_reflectivity_dist

    @bs_reflectivity_dist.setter
    def bs_reflectivity_dist(self, distribution: Distribution) -> None:
        if not isinstance(distribution, Distribution):
            raise TypeError(
                "bs_reflectivity_dist should be a distribution object."
            )
        self._bs_reflectivity_dist = distribution

    @property
    def loss(self) -> int | float:
        """
        Returns a value for loss which depends on the configuration of the error
        model.
        """
        return self._loss_dist.value()

    @property
    def loss_dist(self) -> int | float:
        """Returns currently in use loss value distribution."""
        return self._loss_dist

    @loss_dist.setter
    def loss_dist(self, distribution: Distribution) -> None:
        if not isinstance(distribution, Distribution):
            raise TypeError("loss_dist should be a distribution object.")
        self._loss_dist = distribution

    def _set_random_seed(self, r_seed: int | None) -> None:
        """
        Set the random seed for the error_model to produce repeatable results.
        """
        seed = check_random_seed(r_seed)
        # Set random seed in each property if present
        for prop in [self._bs_reflectivity_dist, self._loss_dist]:
            if hasattr(prop, "set_random_seed") and callable(
                prop.set_random_seed
            ):
                prop.set_random_seed(seed)
