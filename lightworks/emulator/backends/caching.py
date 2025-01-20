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

from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...__settings import settings
from ...sdk.tasks import SamplerTask, TaskData
from ..components import Source
from ..utils import BackendError


@dataclass
class CacheData:
    """
    Stores cached data from the backend.

    Args:

        values: The relevant values used in calculating results for a particular
            cache.

        results: The calculated results dictionary.

    """

    values: list[Any]
    results: dict[Any, Any]


def check_parameter_updates(values1: list[Any], values2: list[Any]) -> bool:
    """
    Determines if parameters have changed between two sets of values.
    """
    for v1, v2 in zip(values1, values2, strict=True):
        # Treat arrays and other values differently
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            if v1.shape != v2.shape:
                return True
            if not (v1 == v2).all():
                return True
        elif isinstance(v1, Source) and isinstance(v2, Source):
            if v1.brightness != v2.brightness:
                return True
            if v1.indistinguishability != v2.indistinguishability:
                return True
            if v1.purity != v2.purity:
                return True
            if v1.probability_threshold != v2.probability_threshold:
                return True
        else:
            if v1 != v2:
                return True
    return False


def get_calculation_values(data: TaskData) -> list[Any]:
    """
    Stores all current parameters used with the sampler in a list and
    returns this.
    """
    if isinstance(data, SamplerTask):
        # Store all values which alter a computation
        return [
            data.circuit.U_full,
            data.circuit.heralds,
            data.input_state,
            copy(data.source),
            copy(settings.sampler_probability_threshold),
        ]
    raise BackendError("Caching not implemented for provided task type.")
