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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import FunctionType, NoneType

from lightworks.sdk.state import State


class PostSelectionType(ABC):
    """
    Base class for post-selection object.
    """

    @abstractmethod
    def validate(self, state: State | list[int]) -> bool:
        """Enforces all post-selection classes have the validate method."""


class PostSelection(PostSelectionType):
    """
    Post-selection

    Args:

        multi_rules (bool, optional) : Controls whether multiple rules can be
            applied to a single mode.

    """

    def __init__(self, multi_rules: bool = False) -> None:
        self.multi_rules = multi_rules
        self.__rules: list[Rule] = []
        self.__modes_with_rules: set[int] = set()

    @property
    def rules(self) -> list["Rule"]:
        """
        Returns currently applied post-selection rules.
        """
        return self.__rules

    @property
    def modes(self) -> list[int]:
        """
        Returns modes on which rules are currently applied.
        """
        return sorted(self.__modes_with_rules)

    def add(
        self, modes: int | tuple[int, ...], n_photons: int | tuple[int, ...]
    ) -> None:
        """
        Adds a post-selection rule across the provided modes, conditioning that
        the total number of photons across the modes is equal to any of the
        provided number of photons.

        Args:

            modes (int, tuple) : The modes across which post-selection should be
                applied. This can be a singular mode provided as an integer or
                a range of modes as a tuple.

            n_photons (int, tuple) : The valid photon number totals across the
                modes. This can be a singular integer value or a range of
                allowable photon numbers.

        """
        modes = check_int_or_tuple(modes)
        n_photons = check_int_or_tuple(n_photons)
        for values, msg in zip(
            (modes, n_photons), ("Mode numbers", "Photon numbers"), strict=True
        ):
            for v in values:
                if v < 0:
                    raise ValueError(msg + " cannot be negative.")
        # Check mode doesn't already have a rule if required
        if not self.multi_rules:
            for m in modes:
                if m in self.__modes_with_rules:
                    msg = (
                        f"Provided mode ({m}) already has a post-selection "
                        "rule applied. To allow this, set multi_rules attribute"
                        " to True."
                    )
                    raise ValueError(msg)
        self.__rules.append(Rule(modes, n_photons))
        for m in modes:
            self.__modes_with_rules.add(m)

    def validate(self, state: State | list[int]) -> bool:
        """
        Validates whether a provided State meets the set post-selection
        criteria.

        Args:

            state (State) : The state which is to be checked.

        Returns:

            bool : Indicates whether the provided State meets the post-selection
                criteria.

        """
        return all(rule.validate(state) for rule in self.__rules)


@dataclass(slots=True)
class Rule:
    """
    Stores a post-selection rule for a number of modes and n_photons.
    """

    modes: tuple[int, ...]
    n_photons: tuple[int, ...]

    def as_tuple(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Returns a tuple representation of the post-selection rule."""
        return (self.modes, self.n_photons)

    def validate(self, state: State | list[int]) -> bool:
        """Validates a provided state meets the post-selection rule."""
        return sum(state[m] for m in self.modes) in self.n_photons


class PostSelectionFunction(PostSelectionType):
    """
    Allows for post-selection to be implemented with a provided function.
    """

    def __init__(self, function: Callable[[State], bool]) -> None:
        if not isinstance(function, FunctionType):
            raise TypeError("Post-selection not a function.")
        self.__function = function

    def validate(self, state: State | list[int]) -> bool:
        """
        Validates whether a provided State meets the set post-selection
        criteria.

        Args:

            state (State) : The state which is to be checked.

        Returns:

            bool : Indicates whether the provided State meets the post-selection
                criteria.

        """
        return self.__function(state)


class DefaultPostSelection(PostSelectionType):
    """
    Provides a default post-selection function which always returns True.
    """

    def validate(self, state: State | list[int]) -> bool:  # noqa: ARG002
        """
        Will return True regardless of provided state.
        """
        return True


def check_int_or_tuple(value: int | Sequence[int]) -> tuple[int, ...]:
    """
    Process a provided value, if it is a sequence then it will check all values
    are integers, otherwise it will validate the value and convert to a tuple.
    """
    if isinstance(value, Sequence):
        value = list(value)
        for i, v in enumerate(value):
            value[i] = check_int(v)
        return tuple(value)
    value = check_int(value)
    return (value,)


def check_int(value: int) -> int:
    """
    Processes a provided single value, trying to convert to an integer if not
    already and and then returning this.
    """
    if not isinstance(value, int):
        if value == int(value):
            value = int(value)
        else:
            raise ValueError(
                "Provided value should be a tuple of integers or an integer."
            )
    return value


def process_post_selection(
    post_selection: PostSelectionType | Callable[[State], bool] | None,
) -> PostSelectionType | None:
    """
    Takes a provided post-selection value and converts this into one of the
    PostSelection objects with a validate method.

    Args:

        post_selection (PostSelectionType | Callable | None) : The
            post-selection value to be processed. If a callable this will be
            converted into the PostSelectionFunction object, otherwise if None
            then the DefaultPostSelection object will be used.

    Returns:

        PostSelectionType | None : The created PostSelection object or None, if
            the original value wasn't one of these types.

    """
    if isinstance(post_selection, FunctionType):
        post_selection = PostSelectionFunction(post_selection)
    elif not isinstance(post_selection, PostSelectionType | NoneType):
        raise TypeError(
            "Provided post_select value should be a PostSelection object, "
            "function or None."
        )
    return post_selection
