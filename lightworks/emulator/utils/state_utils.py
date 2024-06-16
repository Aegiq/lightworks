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

from typing import Iterable

"""
Script to store various useful functions for the simulation aspect of the code.
"""


def fock_basis(N: int, n: int) -> list:  # noqa: N803
    """Returns the Fock basis for n photons in N modes."""
    return list(_sums(N, n))


def _sums(length: int, total_sum: int) -> Iterable:
    if length == 1:
        yield [
            total_sum,
        ]
    else:
        for value in range(total_sum + 1):
            for permutation in _sums(length - 1, total_sum - value):
                yield permutation + [
                    value,
                ]


def annotated_state_to_string(state: list) -> str:
    """Converts the provided annotated state to a string with ket notation."""
    string = "|"
    for s in state:
        if len(s) > 0:
            string += str(len(s)) + ":("
            for label in s:
                string += str(label) + ","
            string = string[:-1] + "),"
        else:
            string += "0,"
    string = string[:-1] + ">"
    return string
