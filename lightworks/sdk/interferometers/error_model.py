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

from ..utils import check_random_seed


class ErrorModel:
    """
    Placeholder class for configuring an error model which can be introduced
    within an interferometer.
    """

    def __init__(self) -> None:
        return

    @property
    def bs_reflectivity(self) -> float:
        """
        Returns a value for beam splitter reflectivity, which depends on the
        configuration of the error model.
        """
        return 0.5

    @property
    def loss(self) -> float:
        """
        Returns a value for loss which depends on the configuration of the error
        model.
        """
        return 0

    def _set_random_seed(self, r_seed: int | None) -> None:
        """
        Set the random seed for the error_model to produce repeatable results.
        """
        seed = check_random_seed(r_seed)
        random.seed(seed)
