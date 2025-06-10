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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lightworks.sdk.utils.exceptions import ModeRangeError

from .photonic_components import Component, Group, HeraldData, Loss


class CompiledPhotonicCircuit:
    """
    Builds up the unitary representation of a particular circuit, combining
    provided components from a circuit spec.

    Args:

        n_modes (int) : The number of modes in a circuit. This should not
            include any required loss modes.

    """

    def __init__(self, n_modes: int) -> None:
        self._n_modes = n_modes
        self._loss_modes = 0
        self._unitary = np.identity(n_modes, dtype=complex)
        self._in_heralds: dict[int, int] = {}
        self._out_heralds: dict[int, int] = {}
        self._circuit_spec: list[tuple[str, dict[str, Any]] | None] = []

    @property
    def n_modes(self) -> int:
        """The number of modes in the circuit."""
        return self._n_modes

    @property
    def loss_modes(self) -> int:
        """The current number of loss modes in the circuit."""
        return self._loss_modes

    @property
    def total_modes(self) -> int:
        """The total number of modes, including real and loss modes."""
        return self.n_modes + self.loss_modes

    @property
    def U_full(self) -> NDArray[np.complex128]:  # noqa: N802
        """Full unitary matrix."""
        return self._unitary

    @property
    def heralds(self) -> HeraldData:
        """Returns details of heralds on the input and output."""
        return HeraldData(
            input=copy(self._in_heralds),
            output=copy(self._out_heralds),
        )

    @property
    def input_modes(self) -> int:
        """
        The number of input modes that should be specified, accounting for the
        heralds used in the circuit.
        """
        return self.n_modes - len(self.heralds.input)

    def add(self, spec: Component) -> None:
        """Adds elements to the spec."""
        if isinstance(spec, Loss):
            self._loss_modes += 1
            self._unitary = np.pad(
                self._unitary, (0, 1), "constant", constant_values=0j
            )
            self._unitary[-1, -1] = 1 + 0j
            self._circuit_spec.append(spec.serialize())
        if isinstance(spec, Group):
            for s in spec.circuit_spec:
                self.add(s)
        else:
            self._unitary = spec.get_unitary(self.total_modes) @ self._unitary
            self._circuit_spec.append(spec.serialize())

    def add_herald(
        self, input_mode: int, output_mode: int, in_photon: int, out_photon: int
    ) -> None:
        """
        Add a herald across a selected input/output of the circuit. If only one
        mode is specified then this will be used for both the input and output.

        Args:

            input_mode (int) : The input mode to use for the herald.

            output_mode (int) : The output mode to use for the herald.

            in_photon (int) : The number of photons to use on the heralding
                input.

            out_photon (int) : The number of photons to use on the heralding
                output.

        """
        for n in [in_photon, out_photon]:
            if not isinstance(n, int) or isinstance(n, bool):
                raise TypeError(
                    "Number of photons for herald should be an integer."
                )
        in_photon = int(in_photon)
        out_photon = int(out_photon)
        if output_mode is None:
            output_mode = input_mode
        self._mode_in_range(input_mode)
        self._mode_in_range(output_mode)
        # Check if herald already used on input or output
        if input_mode in self._in_heralds:
            raise ValueError("Heralding already set for chosen input mode.")
        if output_mode in self._out_heralds:
            raise ValueError("Heralding already set for chosen output mode.")
        # If not then update dictionaries
        self._in_heralds[input_mode] = in_photon
        self._out_heralds[output_mode] = out_photon

    def _mode_in_range(self, mode: int) -> None:
        """
        Check a mode exists within the created circuit and also confirm it
        is an integer.
        """
        if not isinstance(mode, int) or isinstance(mode, bool):
            raise TypeError("Mode number should be an integer.")
        if not 0 <= mode < self.n_modes:
            raise ModeRangeError(
                "Selected mode(s) is not within the range of the created "
                "circuit."
            )
