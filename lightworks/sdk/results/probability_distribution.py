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

from lightworks.sdk.state import State
from lightworks.sdk.utils.exceptions import ProbabilityDistributionError


class ProbabilityDistribution(dict[State, float]):  # noqa: FURB189
    """
    Stores a created ProbabilityDistribution and prevents modification to
    values.
    """

    def __repr__(self) -> str:
        return f"lightworks.ProbabilityDistribution({super().__repr__()})"

    def __setitem__(self, key: State, value: complex) -> None:
        raise ProbabilityDistributionError(
            "Probability distribution should not be modified directly, to edit "
            "this make a copy using dict(ProbabilityDistribution)."
        )
