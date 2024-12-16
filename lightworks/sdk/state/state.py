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
A custom state datatype, which is created with the aim of making states in the
emulator much easier to work with.
"""

from collections.abc import Iterator
from copy import copy
from typing import Any, Union, overload

from ..utils.exceptions import StateError
from .state_utils import state_to_string


class State:
    """
    Custom data type to store information about a quantum state, as well as
    allowing a number of operations to act on the state.

    Args:

        state (list) : The fock basis state to use with the class, this should
            be a list of photon numbers per mode.

    """

    __slots__ = ["__s"]

    def __init__(self, state: list[int]) -> None:
        # If already list then assign to attribute
        if isinstance(state, list):
            self.__s = state
        # Otherwise try to convert
        else:
            self.__s = list(state)
        return

    @property
    def n_photons(self) -> int:
        """Returns the number of photons in a State."""
        return sum(self.__s)

    @property
    def s(self) -> list[int]:
        """Returns a copy of the contents of the state as a list."""
        return copy(self.__s)

    @s.setter
    def s(self, value: Any) -> None:  # noqa: ARG002
        raise StateError("State value should not be modified directly.")

    @property
    def n_modes(self) -> int:
        """The total number of modes in the state."""
        return len(self.__s)

    @n_modes.setter
    def n_modes(self, value: Any) -> None:  # noqa: ARG002
        raise StateError("Number of modes cannot be modified.")

    def merge(self, merge_state: "State") -> "State":
        """Combine two states, summing the number of photons per mode."""
        if self.n_modes != merge_state.n_modes:
            raise ValueError("Merged states must be the same length.")
        return State(
            [n1 + n2 for n1, n2 in zip(self.__s, merge_state.s, strict=True)]
        )

    def _validate(self) -> None:
        """
        Function to perform some validation of a state, including checking that
        the values are all integers and that no negative values are included.
        """
        for s in self.__s:
            if not isinstance(s, int) or isinstance(s, bool):
                raise TypeError(
                    "State mode occupation numbers should be integers."
                )
            if s < 0:
                raise ValueError("Mode occupation numbers cannot be negative.")

    def __str__(self) -> str:
        return state_to_string(self.__s)

    def __repr__(self) -> str:
        return f"lightworks.State({state_to_string(self.__s)})"

    def __add__(self, value: "State") -> "State":
        if not isinstance(value, State):
            raise TypeError("Addition only supported between states.")
        return State(self.__s + value.__s)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, State):
            return False
        return self.__s == value.s

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __len__(self) -> int:
        return self.n_modes

    def __iter__(self) -> Iterator[int]:
        yield from self.s

    def __setitem__(self, key: Any, value: Any) -> None:
        raise StateError("State object does not support item assignment.")

    @overload
    def __getitem__(self, indices: int) -> int: ...

    @overload
    def __getitem__(self, indices: slice) -> "State": ...

    def __getitem__(self, indices: slice | int) -> Union[int, "State"]:
        if isinstance(indices, slice):
            return State(self.__s[indices])
        if isinstance(indices, int):
            return self.__s[indices]
        raise TypeError("Subscript should either be int or slice.")
