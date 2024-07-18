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

from typing import Any

from ..circuit.parameters import Parameter


def process_parameter_value(value: Any, show_parameter_value: bool) -> Any:
    """
    Takes a provided value and will convert into a label or value if it is
    provided as a parameter.
    """
    if not isinstance(value, Parameter):
        return value
    if show_parameter_value:
        return value.get()
    return value.label if value.label is not None else value.get()
