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

from ...sdk.tasks import TaskData
from .caching import CacheData, check_parameter_updates, get_calculation_values

# ruff: noqa: D102


class BackendABC(metaclass=ABCMeta):
    """
    Base class for all backends. An outline of all possible functions should
    be included here.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    def _check_cache(self, data: TaskData) -> dict[str, Any] | None:
        name = data.__class__.__name__
        if hasattr(self, "_cache") and name in self._cache:
            old_values = self._cache[name].values
            new_values = get_calculation_values(data)
            if not check_parameter_updates(old_values, new_values):
                return self._cache[name].results
        # Return false if cache doesn't exist or name not found
        return None

    def _add_to_cache(self, data: TaskData, results: dict[str, Any]) -> None:
        if not hasattr(self, "_cache"):
            self._cache = {}
        name = data.__class__.__name__
        values = get_calculation_values(data)
        self._cache[name] = CacheData(values=values, results=results)
