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

from copy import copy

from lightworks.sdk.state import State


def add_heralds_to_state(
    state: State | list[int], heralds: dict[int, int]
) -> list[int]:
    """
    Takes a provided input state and includes any heralding photons/modes.

    Args:

        state (State | list) : The initial state with heralding modes excluded.

        heralds (dict) : A dictionary of the required heralds to include.

    Returns:

        list : The updated state with heralded modes included.

    """
    # Auto-return original state if no heralding used
    if not heralds:
        return state.s if isinstance(state, State) else copy(state)
    n_modes = len(state) + len(heralds)
    # Otherwise create new state
    new_state = [0] * n_modes
    # Then iterate through modes using values from state or herald
    count = 0
    for i in range(n_modes):
        if i in heralds:
            new_state[i] = heralds[i]
        else:
            new_state[i] = state[count]
            count += 1
    return new_state


def remove_heralds_from_state(
    state: State | list[int], herald_modes: list[int]
) -> list[int]:
    """
    Removes all heralded modes from a provided state.

    Args:

        state (State, list) : The state to remove heralds from.

        herald_modes (list) : A list of the modes used for heralding.

    Returns:

        list : The updated state with heralded modes removed.

    """
    # Remove modes in reverse order so mode locations do not change
    to_remove = sorted(herald_modes, reverse=True)
    # Get list version of state
    new_s = state.s if isinstance(state, State) else copy(state)
    # Then sequentially pop modes
    for m in to_remove:
        new_s.pop(m)
    return new_s
