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
Script to store various useful functions for the simulation aspect of the code.
"""


def state_to_string(state: list[int]) -> str:
    """Converts the provided state to a string with ket notation."""
    string = "|"
    for s in state:
        string += str(s) + ","
    return string[:-1] + ">"
