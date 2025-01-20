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

from collections.abc import Iterable
from math import inf
from numbers import Number
from types import NoneType
from typing import Any

from ..utils import (
    ParameterBoundsError,
    ParameterDictError,
    ParameterValueError,
)


class Parameter:
    """
    Enables the definition a modifiable parameter that can be used as part of a
    Circuit. It allows for the parameter to be modified after utilisation in a
    Circuit for the adjustment of the functionality with having to redefine the
    entire object. Once created the value of the parameter should be modified
    with the get and set functions.

    Args:

        value (Any) : The value to be assigned to the parameter.

        bounds (list | None, optional) : If specified this allows for
            restrictions to be implemented on the value of each parameter. In
            optimisations this can also used to set parameter bounds if the
            optimisation allows. Bounds should be given as a list in the format
            [min_bound, max_bound]. Note bounds are only supported in the case
            of Numeric values.

        label (str | None, optional) : Used to set an optional label which is
            then shown when using the display circuit method, instead of the
            parameter value.

    """

    def __init__(
        self,
        value: Any,
        bounds: list[Number] | None = None,
        label: str | None = None,
    ) -> None:
        # Assign value to attribute
        self.__value = value
        # Process label
        if not isinstance(label, str | NoneType):
            raise TypeError("Label should be a string.")
        self.label = label
        # Process provided bounds
        if not isinstance(bounds, tuple | list | NoneType):
            raise TypeError("Bounds should be a tuple or list.")
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError(
                    "Bounds should have length 2, containing a min and max "
                    "value for the parameter."
                )
            if not isinstance(value, Number) or isinstance(value, bool):
                raise ParameterBoundsError(
                    "Bounds cannot be set for non-numeric parameters."
                )
            self.min_bound, self.max_bound = bounds[0], bounds[1]
        else:
            self.min_bound, self.max_bound = None, None

        return

    @property
    def min_bound(self) -> Number | None:
        """The lower bound of the parameter value."""
        return self.__min_bound

    @min_bound.setter
    def min_bound(self, value: Any) -> None:
        if value is not None:
            if not isnumeric(self.__value):
                raise ParameterBoundsError(
                    "Bounds cannot be set for non-numeric parameters."
                )
            if not isnumeric(value):
                raise ParameterBoundsError("Bound should be numeric or None.")
            if self.__value < value:
                raise ParameterBoundsError(
                    "Current parameter value is below new minimum bound."
                )
        self.__min_bound = value

    @property
    def max_bound(self) -> Number | None:
        """The upper bound of the parameter value."""
        return self.__max_bound

    @max_bound.setter
    def max_bound(self, value: Any) -> None:
        if value is not None:
            if not isnumeric(self.__value):
                raise ParameterBoundsError(
                    "Bounds cannot be set for non-numeric parameters."
                )
            if not isnumeric(value):
                raise ParameterBoundsError("Bound should be numeric or None.")
            if self.__value > value:
                raise ParameterBoundsError(
                    "Current parameter value is above new maximum bound."
                )
        self.__max_bound = value

    def __str__(self) -> str:
        return str(self.__value)

    def __repr__(self) -> str:
        if isinstance(self.__value, str):
            to_output = f"'{self.__value}'"
        else:
            to_output = str(self.__value)
        if self.label is not None:
            to_output += f", '{self.label}'"
        if self.has_bounds():
            to_output += f", [{self.min_bound}, {self.max_bound}]"
        return f"lightworks.Parameter({to_output})"

    def set(self, value: Any) -> None:
        """Update the current value of the parameter."""
        # Don't allow parameter to be set to non-numeric value if bounds set
        if self.has_bounds():  # noqa: SIM102
            if not isinstance(value, Number) or isinstance(value, bool):
                raise ParameterValueError(
                    "Parameter cannot be set to non-numeric value when "
                    "bounds are assigned to parameter."
                )
        if self.min_bound is not None and value < self.min_bound:
            raise ParameterValueError("Set value is below minimum bound.")
        if self.max_bound is not None and value > self.max_bound:
            raise ParameterValueError("Set value is above maximum bound.")
        self.__value = value
        return

    def get(self) -> Any:
        """Returns the current value of the parameter."""
        return self.__value

    def has_bounds(self) -> bool:
        """Checks if the parameter has at least one bound given."""
        return bool(self.min_bound is not None or self.max_bound is not None)


def isnumeric(value: Any) -> bool:
    """General function for checking if a value is numeric."""
    return isinstance(value, Number) and not isinstance(value, bool)


class ParameterDict:
    """
    Stores a number of Parameters, using assigned keys to reference each
    Parameter object. This has custom get and set item which allows for the
    parameter object to be retrieved and the parameter value to be changed with
    the [] operator. For example ParameterDict["a"] would return a Parameter
    object and ParameterDict["a"] = 1 would set the value of the Parameter
    associated with the 'a' key to 1.
    """

    def __init__(self, **kwargs: Parameter) -> None:
        self.__pdict = {}
        for k, v in kwargs.items():
            if not isinstance(v, Parameter):
                self.__pdict[k] = Parameter(v)
            else:
                self.__pdict[k] = v

    @property
    def params(self) -> list[str]:
        """Returns a list of all parameter keys used in the dictionary."""
        return list(self.keys())

    def get_bounds(self) -> dict[str, tuple[Number | float, Number | float]]:
        """
        Retrieves the bounds for all parameters stored in the ParameterDict and
        returns as a dictionary, where the keys match those used to store the
        parameters. If a particular parameter does not have one or both bounds
        set then the bounds will be set to +- inf.
        """
        bounds_dict = {}
        for k, v in self.__pdict.items():
            minb = v.min_bound if v.min_bound is not None else -inf
            maxb = v.max_bound if v.max_bound is not None else inf
            bounds_dict[k] = (minb, maxb)

        return bounds_dict

    def has_bounds(self) -> bool:
        """
        Checks if any of the parameters stored in the dictionary has a minimum
        and/or maximum bounds associated with it.
        """
        # If any parameters has bounds return True, else return False
        return any(v.has_bounds() for v in self.__pdict.values())

    def keys(self) -> Iterable[str]:
        """Returns all keys associated with the Parameters as an iterable."""
        return self.__pdict.keys()

    def items(self) -> list[tuple[str, Any]]:
        """Returns pairs of keys and parameter values in a list."""
        return [(k, v.get()) for k, v in self.__pdict.items()]

    def remove(self, key: Any) -> None:
        """
        Removes the parameter associated with the provided key from the
        ParameterDict.

        Raises:

            KeyError : Raised in cases where the key to remove does not exit in
                the ParameterDict.

        """
        if key not in self.__pdict:
            raise KeyError("Parameter key not found in ParameterDict.")
        del self.__pdict[key]
        return

    def __str__(self) -> str:
        return str(self.__pdict)

    def __repr__(self) -> str:
        return "lightworks.ParameterDict" + str(self.__pdict)

    def __len__(self) -> int:
        return len(self.__pdict)

    def __iter__(self) -> Any:
        """Iterable to allow to do 'for param in ParameterDict'."""
        yield from self.params

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Custom set item behaviors to allow for parameter values to be updated in
        __pdict attribute.
        """
        if key in self.__pdict:
            if isinstance(value, Parameter):
                raise ParameterDictError(
                    "Cannot overwrite existing Parameter with new Parameter "
                    "object."
                )
            self.__pdict[key].set(value)
        else:
            if not isinstance(value, Parameter):
                raise ParameterDictError(
                    "Values being assigned to new keys should be Parameter "
                    "objects."
                )
            self.__pdict[key] = value

    def __getitem__(self, key: str) -> Parameter:
        """Custom get item to return values from __pdict attribute"""
        if key not in self.__pdict:
            msg = f"Parameter '{key}' not found in ParameterDict."
            raise KeyError(msg)
        return self.__pdict[key]

    def __contains__(self, key: Any) -> bool:
        """Custom behaviors for in operator in python."""
        return key in self.__pdict
