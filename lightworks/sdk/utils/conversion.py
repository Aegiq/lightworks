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

from lightworks.sdk.state import State


class convert:  # noqa: N801
    """
    Contains a range of functions for converting/mapping between quantities.
    """

    @staticmethod
    def db_loss_to_decimal(loss: float) -> float:
        """
        Function to convert from a given dB loss into the equivalent loss value
        in decimal form. Note this function does not support conversion of gain
        values.

        Args:

            loss (float) : The loss value in decibels.

        Returns:

            float : The calculated loss as a decimal.

        """
        # Standardize loss format
        loss = -abs(loss)
        return 1 - 10 ** (loss / 10)

    @staticmethod
    def decimal_to_db_loss(loss: float) -> float:
        """
        Function to convert from a decimal into dB loss. This dB loss will be
        returned as a positive value.

        Args:

            loss (float) : The loss value as a decimal, this should be in the
                range [0,1).

        Returns:

            float : The calculated dB loss. This is returned as a positive
                value.

        Raises:

            ValueError: Raised in cases where transmission is not in range
                [0,1).

        """
        if loss < 0 or loss >= 1:
            raise ValueError("Transmission value should be in range [0,1).")
        return abs(10 * log10(1 - loss))

    @staticmethod
    def qubit_to_dual_rail(state: State | list[int] | str) -> State:
        """
        Converts from a qubit encoding into a dual-rail encoded state between
        modes.

        Args:

            state (State) : The qubit state to convert.

        Returns:

            State : The dual-rail encoded Fock state.

        Raises:

            ValueError: Raised when values in the original state aren't either
                0 or 1.

        """
        new_state = []
        for s in state:
            if s in {"0", "1"}:  # Support string values
                s = int(s)  # noqa: PLW2901
            if s not in {0, 1}:
                raise ValueError(
                    "Elements of a qubit state can only take integer values 0 "
                    "or 1."
                )
            new_state += [1, 0] if not s else [0, 1]
        return State(new_state)

    @staticmethod
    def dual_rail_to_qubit(
        state: State | list[int], allow_invalid: bool = False
    ) -> State:
        """
        Converts from a dual-rail encoded Fock state into the qubit encoded
        equivalent.

        Args:

            state (State) : The dual-rail state to convert. This state should
                contain a single photon between pairs of adjacent modes.

            allow_invalid (bool) : Controls whether or not invalid values are
                supported for a qubit. In these cases, the numerical value will
                be replaced by an X.

        Returns:

            State : The calculated qubit state.

        Raises:

            ValueError: Raised when an invalid state is provided for conversion.

        """
        new_state: list[int | str] = []
        if len(state) % 2 != 0:
            raise ValueError(
                "Dual-rail encoded state should have an even number of modes."
            )
        list_state = list(state)
        for i in range(len(state) // 2):
            sub_s = list_state[2 * i : 2 * i + 2]
            if sub_s not in ([1, 0], [0, 1]):
                if not allow_invalid:
                    raise ValueError(
                        "Invalid entry found in state. State should have a "
                        "single photon between each pair of dual-rail encoded "
                        "modes. Alternatively, set allow_invalid = True."
                    )
                new_state.append("X")
            else:
                new_state.append(sub_s[1])
        return State(new_state)  # type: ignore[arg-type]

    @staticmethod
    def threshold_mapping(state: State, invert: bool = False) -> State:
        """
        Apply a threshold mapping to a State, in which any values above 1 will
        be reduced to 1.

        Args:

            state (State) : The state to be converted.

            invert (bool, optional) : Select whether to invert the threshold
                mapping. This will swap the 0s and 1s of the produced states.

        Returns:

            State : The threshold mapped state.

        """
        return State([abs(min(s, 1) - invert) for s in state])

    @staticmethod
    def parity_mapping(state: State, invert: bool = False) -> State:
        """
        Apply a parity mapping to a State, in which the state values are
        converted to either 0 or 1 depending on if the value is odd or even.

        Args:

            state (State) : The state to be converted.

            invert (bool, optional) : Select whether to invert the parity
                mapping. This will swap between even->0 & odd->1 and even->1 &
                odd->0.

        Returns:

            State : The parity mapped state.

        """
        return State([abs((s % 2) - invert) for s in state])
