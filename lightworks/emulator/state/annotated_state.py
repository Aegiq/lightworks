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

"""
A custom state datatype, which is created for storing annotated state details.
It is not intended that this class will be ordinarily accessible to users.
"""

from collections.abc import Iterator
from typing import Any, Union, overload

from lightworks.emulator.utils.exceptions import AnnotatedStateError
from lightworks.emulator.utils.state import annotated_state_to_string


class AnnotatedState:
    """
    Acts as a custom data state which enables fock states to be defined, with
    the main difference to the core State class being that here they are
    defined with a label to indicate photon indistinguishability.

    Args:

        state (list) : The fock basis state to use with the class, this should
            be a list of lists, each containing labels for the photons. The
            number of labels in the list will dictate the total photon number
            in the mode.

    """

    __slots__ = ["__s"]

    def __init__(self, state: list[list[int]]) -> None:
        for s in state:
            if not isinstance(s, list):
                raise TypeError("Provided state labels should be lists.")
        self.__s = [sorted(s) for s in state]

    @property
    def n_photons(self) -> int:
        """Returns the number of photons in a State."""
        return sum(len(s) for s in self.__s)

    @property
    def s(self) -> list[list[int]]:
        """Returns list representation of State."""
        return [list(i) for i in self.__s]

    @s.setter
    def s(self, value: Any) -> None:  # noqa: ARG002
        raise AnnotatedStateError(
            "State value should not be modified directly."
        )

    @property
    def n_modes(self) -> int:
        """Returns total number of modes in the State."""
        return len(self.__s)

    @n_modes.setter
    def n_modes(self, value: Any) -> None:  # noqa: ARG002
        raise AnnotatedStateError("Number of modes cannot be modified.")

    def merge(self, merge_state: "AnnotatedState") -> "AnnotatedState":
        """Combine two states, summing the number of photons per mode."""
        if self.n_modes != merge_state.n_modes:
            raise ValueError("Merged states must be the same length.")
        return AnnotatedState(
            [n1 + n2 for n1, n2 in zip(self.__s, merge_state.s, strict=True)]
        )

    def __str__(self) -> str:
        return annotated_state_to_string(self.__s)

    def __repr__(self) -> str:
        return (
            "lightworks.emulator.state.AnnotatedState("
            f"{annotated_state_to_string(self.__s)})"
        )

    def __add__(self, value: "AnnotatedState") -> "AnnotatedState":
        if not isinstance(value, AnnotatedState):
            raise TypeError("Addition only supported between annotated states.")
        return AnnotatedState(self.__s + value.__s)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AnnotatedState):
            return False
        return self.__s == value.__s

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __len__(self) -> int:
        return self.n_modes

    def __setitem__(self, key: Any, value: Any) -> None:
        raise AnnotatedStateError(
            "AnnotatedState object does not support item assignment."
        )

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.s

    @overload
    def __getitem__(self, indices: int) -> list[int]: ...

    @overload
    def __getitem__(self, indices: slice) -> "AnnotatedState": ...

    def __getitem__(
        self, indices: int | slice
    ) -> Union["AnnotatedState", list[int]]:
        if isinstance(indices, slice):
            return AnnotatedState(self.__s[indices])
        if isinstance(indices, int):
            return self.__s[indices]
        raise TypeError("Subscript should either be int or slice.")
