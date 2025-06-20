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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from lightworks.sdk.state import State
from lightworks.sdk.utils.exceptions import ResultError
from lightworks.sdk.utils.post_selection import PostSelectionType

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

RT = TypeVar("RT")


class Result(dict[_KT, _VT], ABC):
    """
    Base class for all Lightworks result objects.
    """

    @abstractmethod
    def map(
        self: RT,
        mapping: Callable[[State, Any], State],
        *args: Any,
        **kwargs: Any,
    ) -> RT:
        """
        Requires ability to apply a state mapping on a particular result.
        """

    @abstractmethod
    def apply_post_selection(
        self: RT, post_selection: PostSelectionType | Callable[[State], bool]
    ) -> RT:
        """
        Requite the ability to provided additional post-selection criteria to a
        result.
        """

    def __setitem__(self, key: _KT, value: _VT) -> None:
        raise ResultError(
            "Result objects should not be modified directly. To get a copy of "
            "the data which can be modified use dict(Result)."
        )
