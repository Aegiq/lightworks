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
