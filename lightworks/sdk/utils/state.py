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

from lightworks.sdk.circuit import PhotonicCircuit
from lightworks.sdk.state import State

from .exceptions import ModeMismatchError, PhotonNumberError


def validate_states(states: State | list[State], n_modes: int) -> list[State]:
    """
    Performs all required processing/checking a list of states from a task.
    """
    # Convert state to list of States if not provided for single state case
    if isinstance(states, State):
        states = [states]
    # Check each input
    for s in states:
        # Ensure correct type
        if not isinstance(s, State):
            raise TypeError(
                "Assigned states should be a State or list of State objects."
            )
        # Dimension check
        if len(s) != n_modes:
            msg = (
                "One or more states have an incorrect number of modes, correct "
                f"number of modes is {n_modes}."
            )
            raise ModeMismatchError(msg)
        # Also validate state values
        s._validate()
    return states


def check_herald_difference(
    circuit: PhotonicCircuit, input_photons: int
) -> None:
    """
    Validates that the number of output heralds is not too large based on the
    number of inputs to the circuit.
    """
    photon_diff = sum(circuit.heralds.output.values()) - sum(
        circuit.heralds.input.values()
    )
    if photon_diff > input_photons:
        msg = (
            "Number of input photons is smaller than that required based "
            "on the number of heralds at the output. At least "
            f"{photon_diff} input photons are required."
        )
        raise PhotonNumberError(msg)
