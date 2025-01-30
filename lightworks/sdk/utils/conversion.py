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
Contains a number of functions for converting between quantities in the
emulator.
"""

from math import log10

from ..state import State


def db_loss_to_decimal(loss: float) -> float:
    """
    Function to convert from a given dB loss into the equivalent loss value in
    decimal form. Note this function does not support conversion of gain values.

    Args:

        loss (float) : The loss value in decibels.

    Returns:

        float : The calculated loss as a decimal.

    """
    # Standardize loss format
    loss = -abs(loss)
    return 1 - 10 ** (loss / 10)


def decimal_to_db_loss(loss: float) -> float:
    """
    Function to convert from a decimal into dB loss. This dB loss will be
    returned as a positive value.

    Args:

        loss (float) : The loss value as a decimal, this should be in the range
            [0,1).

    Returns:

        float : The calculated dB loss. This is returned as a positive value.

    Raises:

        ValueError: Raised in cases where transmission is not in range [0,1).

    """
    if loss < 0 or loss >= 1:
        raise ValueError("Transmission value should be in range [0,1).")
    return abs(10 * log10(1 - loss))


def qubit_to_dual_rail(state: State) -> State:
    """
    Converts from a qubit encoding into a dual-rail encoded state between modes.

    Args:

        state (State) : The qubit state to convert.

    Returns:

        State : The dual-rail encoded Fock state.

    Raises:

        ValueError: Raised when values in the original state aren't either 0 or
            1.

    """
    new_state = []
    for s in state:
        if s not in (0, 1):
            raise ValueError(
                "Elements of a qubit state can only take integer values 0 or 1."
            )
        new_state += [1, 0] if not s else [0, 1]
    return State(new_state)


def dual_rail_to_qubit(state: State) -> State:
    """
    Converts from a dual-rail encoded Fock state into the qubit encoded
    equivalent.

    Args:

        state (State) : The dual-rail state to convert. This state should
            contain a single photon between pairs of adjacent modes.

    Returns:

        State : The calculated qubit state.

    Raises:

        ValueError: Raised when an invalid state is provided for conversion.

    """
    new_state = []
    if len(state) % 2 != 0:
        raise ValueError(
            "Dual-rail encoded state should have an even number of modes."
        )
    list_state = list(state)
    for i in range(len(state) // 2):
        sub_s = list_state[2 * i : 2 * i + 2]
        if sub_s not in ([1, 0], [0, 1]):
            raise ValueError(
                "Invalid entry found in state. State should have a single "
                "photon between each pair of dual-rail encoded modes."
            )
        new_state.append(sub_s[1])
    return State(new_state)
