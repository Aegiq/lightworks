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

from enum import Enum
from typing import TypeVar, overload

from lightworks.sdk.circuit.parameters import Parameter

T = TypeVar("T")


@overload
def process_parameter_value(
    value: Parameter[T], show_parameter_value: bool
) -> str | T: ...


@overload
def process_parameter_value(value: T, show_parameter_value: bool) -> T: ...


def process_parameter_value(
    value: T | Parameter[T], show_parameter_value: bool
) -> str | T:
    """
    Takes a provided value and will convert into a label or value if it is
    provided as a parameter.
    """
    if not isinstance(value, Parameter):
        return value
    if show_parameter_value:
        return value.get()
    return value.label if value.label is not None else value.get()


class MPLSettings(Enum):
    """Contains settings for the matplotlib display method."""

    TEXT_SIZE = 10
    S_TEXT_SIZE = 7


class SVGSettings(Enum):
    """Contains settings for the drawsvg display method."""

    TEXT_SIZE = 25
    U_TEXT_SIZE = 20
    S_TEXT_SIZE = 18
