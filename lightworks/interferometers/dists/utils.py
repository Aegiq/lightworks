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


def is_number(value: Any | list[Any]) -> None:
    """
    Function to check if the provided value is float or integer.
    """
    if not isinstance(value, list | tuple):
        value = [value]
    for v in value:
        if not isinstance(v, int | float) or isinstance(v, bool):
            raise TypeError("Distribution values should be a float or integer.")
