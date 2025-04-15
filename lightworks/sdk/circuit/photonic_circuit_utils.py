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

# ruff: noqa: PLW2901

"""
Contains a number of different utility functions for modifying circuits.
"""

from copy import copy
from numbers import Real
from typing import Any

import numpy as np
import sympy as sp
from multimethod import multimethod
from numpy.typing import NDArray

from lightworks.sdk.utils.param_unitary import ParameterizedUnitary

from .parameters import Parameter
from .photonic_components import (
    Barrier,
    BeamSplitter,
    Component,
    Group,
    HeraldData,
    Loss,
    ModeSwaps,
    PhaseShifter,
    UnitaryMatrix,
)


def unpack_circuit_spec(circuit_spec: list[Component]) -> list[Component]:
    """
    Unpacks and removes any grouped components from a circuit.

    Args:

        circuit_spec (list) : The circuit spec to unpack.

    Returns:

        list : The processed circuit spec.

    """
    new_spec = list(circuit_spec)
    while any(isinstance(s, Group) for s in new_spec):
        temp_spec = []
        for spec in circuit_spec:
            if not isinstance(spec, Group):
                temp_spec += [spec]
            else:
                temp_spec += spec.circuit_spec
        new_spec = temp_spec

    return new_spec


def add_modes_to_circuit_spec(
    circuit_spec: list[Component], mode: int
) -> list[Component]:
    """
    Takes an existing circuit spec and adds a given number of modes to each
    of the elements.

    Args:

        circuit_spec (list) : The circuit spec which is to be modified.

        mode (int) : The number of modes to shift each of the elements by.

    Returns:

        list : The modified version of the circuit spec.

    """
    new_circuit_spec = []
    for spec in circuit_spec:
        spec = copy(spec)
        if isinstance(spec, BeamSplitter):
            spec.mode_1 += mode
            spec.mode_2 += mode
        elif isinstance(spec, Barrier):
            spec.modes = [p + mode for p in spec.modes]
        elif isinstance(spec, ModeSwaps):
            spec.swaps = {k + mode: v + mode for k, v in spec.swaps.items()}
        elif isinstance(spec, Group):
            spec.circuit_spec = add_modes_to_circuit_spec(
                spec.circuit_spec, mode
            )
            spec.mode_1 += mode
            spec.mode_2 += mode
        else:
            spec.mode += mode  # type: ignore[attr-defined]
        new_circuit_spec.append(spec)
    return new_circuit_spec


def add_empty_mode_to_circuit_spec(
    circuit_spec: list[Component], mode: int
) -> list[Component]:
    """
    Takes a provided circuit spec and adds an empty mode at the set location.

    Args:

        circuit_spec (list) : The circuit spec which is to be modified.

        mode (int) : The location at which an empty mode should be included.

    Returns:

        list : The modified version of the circuit spec.

    """
    new_circuit_spec = []
    for spec in circuit_spec:
        spec = copy(spec)
        if isinstance(spec, BeamSplitter):
            spec.mode_1 += 1 if spec.mode_1 >= mode else 0
            spec.mode_2 += 1 if spec.mode_2 >= mode else 0
        elif isinstance(spec, Barrier):
            spec.modes = [p + 1 if p >= mode else p for p in spec.modes]
        elif isinstance(spec, ModeSwaps):
            swaps = {}
            for k, v in spec.swaps.items():
                k += 1 if k >= mode else 0
                v += 1 if v >= mode else 0
                swaps[k] = v
            spec.swaps = swaps
        elif isinstance(spec, Group):
            spec.circuit_spec = add_empty_mode_to_circuit_spec(
                spec.circuit_spec, mode
            )
            # Shift unitary mode range
            spec.mode_1 += 1 if spec.mode_1 >= mode else 0
            spec.mode_2 += 1 if spec.mode_2 >= mode else 0
            # Update herald values
            new_in_heralds = {}
            for m, n in spec.heralds.input.items():
                if m >= (mode - spec.mode_1) and mode - spec.mode_1 >= 0:
                    m += 1
                new_in_heralds[m] = n
            new_out_heralds = {}
            for m, n in spec.heralds.output.items():
                if m >= (mode - spec.mode_1) and mode - spec.mode_1 >= 0:
                    m += 1
                new_out_heralds[m] = n
            spec.heralds = HeraldData(
                input=new_in_heralds, output=new_out_heralds
            )
        elif isinstance(spec, UnitaryMatrix):
            spec.mode += 1 if spec.mode >= mode else 0
            # Expand unitary if required
            if spec.mode < mode < spec.mode + spec.unitary.shape[0]:
                add_mode = mode - spec.mode
                # Update unitary value
                spec.unitary = add_mode_to_unitary(spec.unitary, add_mode)
        else:
            spec.mode += 1 if spec.mode >= mode else 0  # type: ignore[attr-defined]
        new_circuit_spec.append(spec)
    return new_circuit_spec


def convert_non_adj_beamsplitters(
    circuit_spec: list[Component],
) -> list[Component]:
    """
    Takes a given circuit spec and removes all beam splitters acting on
    non-adjacent modes by replacing with a mode swap and adjacent beam
    splitters.

    Args:

        circuit_spec (list) : The circuit spec to remove beam splitter on
            non-adjacent modes from.

    Returns:

        list : The processed circuit spec.

    """
    new_spec: list[Component] = []
    for spec in circuit_spec:
        spec = copy(spec)
        if (
            isinstance(spec, BeamSplitter)
            and abs(spec.mode_2 - spec.mode_1) != 1
        ):
            m1, m2 = spec.mode_1, spec.mode_2
            if m1 > m2:
                m1, m2 = m2, m1
            mid = int((m1 + m2 - 1) / 2)
            swaps = {}
            for i in range(m1, mid + 1):
                swaps[i] = mid if i == m1 else i - 1
            for i in range(mid + 1, m2 + 1):
                swaps[i] = mid + 1 if i == m2 else i + 1
            new_spec.append(ModeSwaps(swaps))
            # If original modes were inverted then invert here too
            add1, add2 = mid, mid + 1
            if spec.mode_1 > spec.mode_2:
                add1, add2 = add2, add1
            # Add beam splitter on new modes
            new_spec.append(
                BeamSplitter(add1, add2, spec.reflectivity, spec.convention)
            )
            swaps = {v: k for k, v in swaps.items()}
            new_spec.append(ModeSwaps(swaps))
        elif isinstance(spec, Group):
            spec.circuit_spec = convert_non_adj_beamsplitters(spec.circuit_spec)
            new_spec.append(spec)
        else:
            new_spec.append(spec)
    return new_spec


def compress_mode_swaps(circuit_spec: list[Component]) -> list[Component]:
    """
    Takes a provided circuit spec and will try to compress any more swaps
    such that the circuit length is reduced. Note that any components in a
    group will be ignored.

    Args:

        circuit_spec (list) : The circuit spec which is to be processed.

    Returns:

        list : The processed version of the circuit spec.

    """
    new_spec: list[Component] = []
    to_skip: list[int] = []
    # Loop over each item in original spec
    for i, spec in enumerate(circuit_spec):
        spec = copy(spec)
        if i in to_skip:
            continue
        # If it a mode swap then check for subsequent mode swaps
        if isinstance(spec, ModeSwaps):
            blocked_modes = set()
            for j, spec2 in enumerate(circuit_spec[i + 1 :]):
                # Block modes with components other than the mode swap on
                if isinstance(spec2, PhaseShifter | Loss):
                    # NOTE: In principle a phase shift doesn't need to
                    # block a mode and instead we could modify it's
                    # location
                    blocked_modes.add(spec2.mode)
                elif isinstance(spec2, BeamSplitter):
                    blocked_modes.add(spec2.mode_1)
                    blocked_modes.add(spec2.mode_2)
                elif isinstance(spec2, Group):
                    blocked_modes.update(range(spec2.mode_1, spec2.mode_2 + 1))
                elif isinstance(spec2, UnitaryMatrix):
                    blocked_modes.update(
                        range(spec2.mode, spec2.mode + spec2.unitary.shape[0])
                    )
                elif isinstance(spec2, ModeSwaps):
                    # When a mode swap is found check if any of its mode
                    # are in the blocked mode
                    swaps = spec2.swaps
                    for m in swaps:
                        # If they are then block all other modes of swap
                        if m in blocked_modes:
                            blocked_modes.update(swaps)
                            break
                    else:
                        # Otherwise combine the original and found swap
                        # and update spec entry
                        new_swaps = combine_mode_swap_dicts(spec.swaps, swaps)
                        spec.swaps = new_swaps
                        # Also set to skip the swap that was combine
                        to_skip.append(i + 1 + j)
            new_spec.append(spec)
        else:
            new_spec.append(spec)

    return new_spec


def combine_mode_swap_dicts(
    swaps1: dict[int, int], swaps2: dict[int, int]
) -> dict[int, int]:
    """
    Function to take two mode swap dictionaries and combine them.

    Args:

        swaps1 (dict) : The first mode swap dictionary to combine.

        swaps2 (dict) : The mode swap dictionary to combine with the first
            dictionary.

    Returns:

        dict : The calculated combined mode swap dictionary.

    """
    # Store overall swaps in new dictionary
    new_swaps = {}
    added_swaps = []
    for k1, v1 in swaps1.items():
        for k2, v2 in swaps2.items():
            # Loop over swaps to combine when a key from swap 2 is in the
            # values of swap 1
            if v1 == k2:
                new_swaps[k1] = v2
                added_swaps.append(k2)
                break
        # If it isn't found then add key and value from swap 1
        else:
            new_swaps[k1] = v1
    # Add any keys from swaps2 that weren't used
    new_swaps.update({k: v for k, v in swaps2.items() if k not in added_swaps})
    # Remove any modes that are unchanged as these are not required
    return {m1: m2 for m1, m2 in new_swaps.items() if m1 != m2}


@multimethod
def check_loss(loss: Any) -> None:  # noqa: ARG001
    """
    Performs validation that a provided loss value is valid.
    """
    # Generic case to catch when an incorrect type is provided.
    raise TypeError(
        "Loss value should be a real numerical value or a Parameter. If set as "
        "a parameter then ensure the parameter value is a real number."
    )


@check_loss.register
def _check_loss_real(loss: Real) -> None:
    """
    Check that loss is assigned to a numerical value in the correct range.
    """
    if isinstance(loss, bool):
        raise TypeError("Loss value cannot be a boolean.")
    if not 0.0 <= float(loss) <= 1.0:
        raise ValueError("Provided loss values should be in the range [0,1].")


@check_loss.register
def _check_loss_param(loss: Parameter[Any]) -> None:
    """Check that loss value is valid when it is assigned to a parameter."""
    return check_loss(loss.get())


@multimethod
def add_mode_to_unitary(
    unitary: np.ndarray[Any, Any], add_mode: int
) -> NDArray[np.complex128]:
    """
    Adds a new mode (through inclusion of an extra row/column) to the provided
    unitary at the selected location.

    Args:

        unitary (np.ndarray | ParameterizedUnitary) : The unitary to add a mode
            to.

        add_mode (int) : The location at which a new mode should be added to
            the circuit.

    Returns:

        np.ndarray | ParameterizedUnitary : The converted unitary matrix.

    """
    dim = unitary.shape[0] + 1
    new_u = np.identity(dim, dtype=complex)
    # Diagonals
    new_u[:add_mode, :add_mode] = unitary[:add_mode, :add_mode]
    new_u[add_mode + 1 :, add_mode + 1 :] = unitary[add_mode:, add_mode:]
    # Off-diagonals
    new_u[:add_mode, add_mode + 1 :] = unitary[:add_mode, add_mode:]
    new_u[add_mode + 1 :, :add_mode] = unitary[add_mode:, :add_mode]
    return new_u


@add_mode_to_unitary.register
def _add_mode_to_parameterized_unitary(
    unitary: ParameterizedUnitary, add_mode: int
) -> ParameterizedUnitary:
    """Adds mode at selected location for ParameterizedUnitary."""
    dim = unitary.shape[0] + 1
    new_u = sp.Matrix([[float(i == j) for j in range(dim)] for i in range(dim)])
    # Diagonals
    new_u[:add_mode, :add_mode] = unitary._unitary[:add_mode, :add_mode]
    new_u[add_mode + 1 :, add_mode + 1 :] = unitary._unitary[
        add_mode:, add_mode:
    ]
    # Off-diagonals
    new_u[:add_mode, add_mode + 1 :] = unitary._unitary[:add_mode, add_mode:]
    new_u[add_mode + 1 :, :add_mode] = unitary._unitary[add_mode:, :add_mode]
    return ParameterizedUnitary(new_u, unitary._params)


def find_optimal_mode_swapping(
    required_swaps: dict[int, int], n_modes: int
) -> dict[int, int]:
    """
    Finds the optimal mode swapping (i.e. minimising swaps) for a particular set
    of required swaps by mapping to the first available mode.

    Args:

        required_swaps (dict) : The swaps that need to be realised by the
            operation.

        n_modes (int) : The number of modes across which swapping occurs.

    Returns:

        dict : The calculated swap dictionary.

    """
    current_mode = 0
    swaps = {}
    # Loop over all modes in circuit to find swaps
    for i in range(n_modes):
        # If used as a key then take value from provisional swaps
        if i in required_swaps:
            swaps[i] = required_swaps[i]
        # Otherwise then map mode to lowest mode possible
        else:
            while current_mode in required_swaps.values():
                current_mode += 1
            if i != current_mode:
                swaps[i] = current_mode
            current_mode += 1
    return swaps
