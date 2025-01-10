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
PhotonicCircuit class for creating circuits with Parameters object that can be
modified after creation.
"""

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Union

import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from numpy.typing import NDArray

from ..utils import (
    CircuitCompilationError,
    ModeRangeError,
)
from ..visualisation import Display
from .circuit_utils import (
    add_empty_mode_to_circuit_spec,
    add_modes_to_circuit_spec,
    check_loss,
    compress_mode_swaps,
    convert_non_adj_beamsplitters,
    unpack_circuit_spec,
)
from .parameters import Parameter
from .photonic_compiler import CompiledPhotonicCircuit
from .photonic_components import (
    Barrier,
    BeamSplitter,
    Component,
    Group,
    Loss,
    ModeSwaps,
    PhaseShifter,
)

if TYPE_CHECKING:
    from .unitary import Unitary


class PhotonicCircuit:
    """
    Provides support for building photonic circuits from a set of linear optic
    components, with the ability to assign certain quantities of components to
    Parameter objects whose values can be adjusted after creation.

    Args:

        n_modes (int) : The number of modes used in the circuit.

    """

    def __init__(self, n_modes: int) -> None:
        if not isinstance(n_modes, int):
            if int(n_modes) == n_modes:
                n_modes = int(n_modes)
            else:
                raise TypeError("Number of modes should be an integer.")
        self.__n_modes = n_modes
        self.__circuit_spec: list[Component] = []
        self.__in_heralds: dict[int, int] = {}
        self.__out_heralds: dict[int, int] = {}
        self.__external_in_heralds: dict[int, int] = {}
        self.__external_out_heralds: dict[int, int] = {}
        self.__internal_modes: list[int] = []

    def __add__(self, value: "PhotonicCircuit") -> "PhotonicCircuit":
        """Defines custom addition behaviour for two circuits."""
        if not isinstance(value, PhotonicCircuit):
            raise TypeError(
                "Addition only supported between two PhotonicCircuit objects."
            )
        if self.n_modes != value.n_modes:
            raise ModeRangeError(
                "Two circuits to add must have the same number of modes."
            )
        if self.heralds["input"] or value.heralds["input"]:
            raise NotImplementedError(
                "Support for heralds when combining circuits not yet "
                "implemented."
            )
        # Create new circuits and combine circuits specs to create new one
        new_circ = PhotonicCircuit(self.n_modes)
        new_circ.__circuit_spec = self.__circuit_spec + value.__circuit_spec
        return new_circ

    def __str__(self) -> str:
        return f"PhotonicCircuit({self.n_modes})"

    def __repr__(self) -> str:
        return f"lightworks.PhotonicCircuit({self.n_modes})"

    @property
    def U(self) -> NDArray[np.complex128]:  # noqa: N802
        """
        The effective unitary that the circuit implements across modes. This
        will include the effect of any loss within a circuit. It is calculated
        using the parameter values at the time that the attribute is called.
        """
        return self._build().U_full[: self.n_modes, : self.n_modes]

    @property
    def U_full(self) -> NDArray[np.complex128]:  # noqa: N802
        """
        The full unitary for the created circuit, this will include the
        additional modes used for the simulation of loss, if this has been
        included in the circuit.
        """
        return self._build().U_full

    @property
    def n_modes(self) -> int:
        """The number of modes in the circuit."""
        return self.__n_modes

    @n_modes.setter
    def n_modes(self, value: Any) -> None:  # noqa: ARG002
        """
        Prevents modification of n_modes attribute after circuit creation.
        """
        raise AttributeError("Number of circuit modes cannot be modified.")

    @property
    def input_modes(self) -> int:
        """
        The number of input modes that should be specified, accounting for the
        heralds used in the circuit.
        """
        return self.n_modes - len(self.heralds["input"])

    @property
    def heralds(self) -> dict[str, dict[int, int]]:
        """
        A dictionary which details the set heralds on the inputs and outputs of
        the circuit.
        """
        return {
            "input": copy(self.__in_heralds),
            "output": copy(self.__out_heralds),
        }

    @property
    def _internal_modes(self) -> list[int]:
        return self.__internal_modes

    @property
    def _external_heralds(self) -> dict[str, dict[int, int]]:
        """
        Stores details of heralds which are on the outside of a circuit (i.e.
        were not added as part of a group).
        """
        return {
            "input": copy(self.__external_in_heralds),
            "output": copy(self.__external_out_heralds),
        }

    def add(
        self,
        circuit: Union["PhotonicCircuit", "Unitary"],
        mode: int = 0,
        group: bool = False,
        name: str | None = None,
    ) -> None:
        """
        Can be used to add either another PhotonicCircuit or a Unitary component
        to the existing circuit. This can either have the same size or be
        smaller than the circuit which is being added to.

        Args:

            circuit (PhotonicCircuit | Unitary) : The circuit/component that is
                to be added.

            mode (int, optional) : The first mode on which the circuit should
                be placed. If not specified it defaults to zero.

            group (bool, optional) : Used to control whether the circuit
                components are added to the existing circuit or placed within a
                group which contains all components in a single element.
                Defaults to False unless the added circuit has heralds, in
                which grouping is enforced.

            name (str | None, optional) : Set a name to use when displaying the
                added circuit. This is only applied when the group option is
                used.

        """
        if not isinstance(circuit, PhotonicCircuit):
            raise TypeError(
                "Add method only supported for PhotonicCircuit or Unitary "
                "objects."
            )
        # Remap mode
        mode = self._map_mode(mode)
        self._mode_in_range(mode)
        # Make copy of circuit to avoid modification
        circuit_copy = circuit.copy()
        # Use unpack groups and check if heralds are used
        circuit_copy.unpack_groups()
        # Force grouping if heralding included
        group = True if circuit_copy.heralds["input"] else group
        # When name not provided set this
        if name is None:
            spec = circuit.__circuit_spec
            # Check special case where name is retrieved from previous group
            name = (
                spec[0].name
                if len(spec) == 1 and isinstance(spec[0], Group)
                else "Circuit"
            )
        # When grouping use unpacked circuit
        if group:
            circuit = circuit_copy
        spec = circuit.__circuit_spec
        # Check circuit size is valid
        n_heralds = len(circuit.heralds["input"])
        if mode + circuit.n_modes - n_heralds > self.n_modes:
            raise ModeRangeError("Circuit to add is outside of mode range")

        # Include any existing internal modes into the circuit to be added
        for i in sorted(self.__internal_modes):
            # Need to account for shifts when adding new heralds
            target_mode = i - mode
            for m in circuit.heralds["input"]:
                if target_mode > m:
                    target_mode += 1
            if 0 <= target_mode < circuit.n_modes:
                spec = circuit._add_empty_mode(spec, target_mode)
        # Then add new modes for heralds from circuit and also add swaps to
        # enforce that the input and output herald are on the same mode
        provisional_swaps = {}
        for m in sorted(circuit.heralds["input"]):
            self.__circuit_spec = self._add_empty_mode(
                self.__circuit_spec, mode + m
            )
            self.__internal_modes.append(mode + m)
            # Current limitation is that heralding should be on the same mode
            # when adding, so use a mode swap to compensate for this.
            herald_loc = list(circuit.heralds["input"].keys()).index(m)
            out_herald = list(circuit.heralds["output"].keys())[herald_loc]
            provisional_swaps[out_herald] = m
        # Convert provisional swaps into full list and add to circuit
        current_mode = 0
        swaps = {}
        # Loop over all modes in circuit to find swaps
        for i in range(circuit.n_modes):
            # If used as a key then take value from provisional swaps
            if i in provisional_swaps:
                swaps[i] = provisional_swaps[i]
            # Otherwise then map mode to lowest mode possible
            else:
                while current_mode in provisional_swaps.values():
                    current_mode += 1
                if i != current_mode:
                    swaps[i] = current_mode
                current_mode += 1
        # Skip for cases where swaps do not alter mode structure
        if list(swaps.keys()) != list(swaps.values()):
            spec.append(ModeSwaps(swaps))
        # Update heralds to enforce input and output are on the same mode
        new_heralds = {
            "input": circuit.heralds["input"],
            "output": circuit.heralds["input"],
        }
        # Also add all included heralds to the heralds dict
        for m in new_heralds["input"]:
            self.__in_heralds[m + mode] = new_heralds["input"][m]
            self.__out_heralds[m + mode] = new_heralds["input"][m]
        # And shift all components in circuit by required amount
        add_cs = add_modes_to_circuit_spec(spec, mode)

        # Then add circuit spec, adjusting how this is included
        if not group:
            self.__circuit_spec = self.__circuit_spec + add_cs
        else:
            self.__circuit_spec.append(
                Group(
                    add_cs, name, mode, mode + circuit.n_modes - 1, new_heralds
                )
            )
        return

    def bs(
        self,
        mode_1: int,
        mode_2: int | None = None,
        reflectivity: float = 0.5,
        loss: float = 0,
        convention: str = "Rx",
    ) -> None:
        """
        Add a beam splitter of specified reflectivity between two modes to the
        circuit.

        Args:

            mode_1 (int) : The first mode which the beam splitter acts on.

            mode_2 (int | None, optional) : The second mode that the beam
                splitter acts on. This can also be left as the default value of
                None to automatically use mode_2 as mode_1 + 1.

            reflectivity (float, optional) : The reflectivity value of the
                beam splitter. Defaults to 0.5.

            loss (float, optional) : The loss of the beam splitter, this should
                be provided as a decimal value in the range [0,1].

            convention (str, optional) : The convention to use for the beam
                splitter, should be either "Rx" (the default value) or "H".

        """
        if mode_2 is None:
            mode_2 = mode_1 + 1
        mode_1 = self._map_mode(mode_1)
        self._mode_in_range(mode_1)
        mode_2 = self._map_mode(mode_2)
        if mode_1 == mode_2:
            raise ModeRangeError(
                "Beam splitter must act across two different modes."
            )
        self._mode_in_range(mode_2)
        # Validate loss before updating circuit spec
        check_loss(loss)
        # Then update circuit spec
        self.__circuit_spec.append(
            BeamSplitter(mode_1, mode_2, reflectivity, convention)
        )
        if isinstance(loss, Parameter) or loss > 0:
            self.loss(mode_1, loss)
            self.loss(mode_2, loss)

    def ps(self, mode: int, phi: float, loss: float = 0) -> None:
        """
        Applies a phase shift to a given mode in the circuit.

        Args:

            mode (int) : The mode on which the phase shift acts.

            phi (float) : The angular phase shift to apply.

            loss (float, optional) : The loss of the phase shifter, this should
                be provided as a decimal value in the range [0,1].

        """
        mode = self._map_mode(mode)
        self._mode_in_range(mode)
        check_loss(loss)
        self.__circuit_spec.append(PhaseShifter(mode, phi))
        if isinstance(loss, Parameter) or loss > 0:
            self.loss(mode, loss)

    def loss(self, mode: int, loss: float = 0) -> None:
        """
        Adds a loss channel to the specified mode of the circuit.

        Args:

            mode (int) : The mode on which the loss channel acts.

            loss (float, optional) : The loss applied to the selected mode,
                this should be provided as a decimal value in the range [0,1].

        """
        mode = self._map_mode(mode)
        self._mode_in_range(mode)
        check_loss(loss)
        self.__circuit_spec.append(Loss(mode, loss))

    def barrier(self, modes: list[int] | None = None) -> None:
        """
        Adds a barrier to separate different parts of a circuit. This is
        applied to the specified modes.

        Args:

            modes (list | None) : The modes over which the barrier should be
                applied to.

        """
        if modes is None:
            modes = list(range(self.n_modes - len(self.__internal_modes)))
        modes = [self._map_mode(i) for i in modes]
        for m in modes:
            self._mode_in_range(m)
        self.__circuit_spec.append(Barrier(modes))

    def mode_swaps(self, swaps: dict[int, int]) -> None:
        """
        Performs swaps between a given set of modes. The keys of the dictionary
        should correspond to the initial modes and the values to the modes they
        are swapped to. It is also required that all mode swaps are complete,
        i.e. any modes which are swapped to must also be sent to a target
        destination.

        Args:

            swaps (dict) : A dictionary detailing the original modes and the
                locations that they are to be swapped to.

        """
        # Remap swap dict and check modes
        swaps = {
            self._map_mode(mi): self._map_mode(mo) for mi, mo in swaps.items()
        }
        for m in [*swaps.keys(), *swaps.values()]:
            self._mode_in_range(m)
        self.__circuit_spec.append(ModeSwaps(swaps))

    def herald(
        self, n_photons: int, input_mode: int, output_mode: int | None = None
    ) -> None:
        """
        Add a herald across a selected input/output of the circuit. If only one
        mode is specified then this will be used for both the input and output.

        Args:

            n_photons (int) : The number of photons to use for the heralding.

            input_mode (int) : The input mode to use for the herald.

            output_mode (int | None, optional) : The output mode for the
                herald, if this is not specified it will be set to the value of
                the input mode.

        """
        if not isinstance(n_photons, int) or isinstance(n_photons, bool):
            raise TypeError(
                "Number of photons for herald should be an integer."
            )
        n_photons = int(n_photons)
        if output_mode is None:
            output_mode = input_mode
        input_mode = self._map_mode(input_mode)
        output_mode = self._map_mode(output_mode)
        self._mode_in_range(input_mode)
        self._mode_in_range(output_mode)
        # Check if herald already used on input or output
        if input_mode in self.__in_heralds:
            raise ValueError("Heralding already set for chosen input mode.")
        if output_mode in self.__out_heralds:
            raise ValueError("Heralding already set for chosen output mode.")
        # If not then update dictionaries
        self.__in_heralds[input_mode] = n_photons
        self.__out_heralds[output_mode] = n_photons
        self.__external_in_heralds[input_mode] = n_photons
        self.__external_out_heralds[output_mode] = n_photons

    def display(
        self,
        show_parameter_values: bool = False,
        display_loss: bool = False,
        mode_labels: list[str] | None = None,
        display_type: str = "svg",
    ) -> None:
        """
        Displays the current circuit with parameters set using either their
        current values or labels.

        Args:

            show_parameter_values (bool, optional) : Shows the values of
                parameters instead of the associated labels if specified.

            display_loss (bool, optional) : Choose whether to display loss
                components in the figure, defaults to False.

            mode_labels (list|None, optional) : Optionally provided a list of
                mode labels which will be used to name the mode something other
                than numerical values. Can be set to None to use default
                values.

            display_type (str, optional) : Selects whether the drawsvg or
                matplotlib module should be used for displaying the circuit.
                Should either be 'svg' or 'mpl', defaults to 'svg'.

        """
        return_ = Display(
            self,
            display_loss=display_loss,
            mode_labels=mode_labels,
            display_type=display_type,
            show_parameter_values=show_parameter_values,
        )
        if display_type == "mpl":
            plt.show()
        elif display_type == "svg":
            display.display(return_)
        return

    def get_all_params(self) -> list[Parameter]:
        """
        Returns all the Parameter objects used as part of creating the circuit.
        """
        all_params = []
        for spec in unpack_circuit_spec(self.__circuit_spec):
            for p in spec.values():
                if isinstance(p, Parameter) and p not in all_params:
                    all_params.append(p)
        return all_params

    def copy(self, freeze_parameters: bool = False) -> "PhotonicCircuit":
        """
        Creates and returns an identical copy of the circuit, optionally
        freezing parameter values.

        Args:

            freeze_parameters (bool, optional) : Used to control where any
                existing parameter objects are carried over to the newly
                created circuit, or if the current parameter values should be
                used. Defaults to False.

        Returns:

            PhotonicCircuit : A new PhotonicCircuit object with the same
                circuit configuration as the original object.

        """
        new_circ = PhotonicCircuit(self.n_modes)
        if not freeze_parameters:
            new_circ.__circuit_spec = copy(self.__circuit_spec)
        else:
            copied_spec = deepcopy(self.__circuit_spec)
            new_circ.__circuit_spec = list(self._freeze_params(copied_spec))
        new_circ.__in_heralds = copy(self.__in_heralds)
        new_circ.__out_heralds = copy(self.__out_heralds)
        new_circ.__external_in_heralds = copy(self.__external_in_heralds)
        new_circ.__external_out_heralds = copy(self.__external_out_heralds)
        new_circ.__internal_modes = copy(self.__internal_modes)
        return new_circ

    def unpack_groups(self) -> None:
        """
        Unpacks any component groups which may have been added to the circuit.
        """
        self.__internal_modes = []
        self.__external_in_heralds = self.__in_heralds
        self.__external_out_heralds = self.__out_heralds
        self.__circuit_spec = unpack_circuit_spec(self.__circuit_spec)
        return

    def compress_mode_swaps(self) -> None:
        """
        Takes a provided circuit spec and will try to compress any more swaps
        such that the circuit length is reduced. Note that any components in a
        group will be ignored.
        """
        # Convert circuit spec and then assign to attribute
        new_spec = compress_mode_swaps(deepcopy(self.__circuit_spec))
        self.__circuit_spec = new_spec

    def remove_non_adjacent_bs(self) -> None:
        """
        Removes any beam splitters acting on non-adjacent modes by replacing
        with a mode swap and adjacent beam splitters.
        """
        # Convert circuit spec and then assign to attribute
        spec = deepcopy(self.__circuit_spec)
        new_spec = convert_non_adj_beamsplitters(spec)
        self.__circuit_spec = new_spec

    def _build(self) -> CompiledPhotonicCircuit:
        """
        Converts the ParameterizedCircuit into a circuit object using the
        components added and current values of the parameters.
        """
        try:
            circuit = self._build_process()
        except Exception as e:
            msg = "An error occurred during the circuit compilation process"
            raise CircuitCompilationError(msg) from e

        return circuit

    def _build_process(self) -> CompiledPhotonicCircuit:
        """
        Contains full process for convert a circuit into a compiled one.
        """
        circuit = CompiledPhotonicCircuit(self.n_modes)

        for spec in self.__circuit_spec:
            circuit.add(spec)

        heralds = self.heralds
        for i, o in zip(heralds["input"], heralds["output"], strict=True):
            circuit.add_herald(heralds["input"][i], i, o)

        return circuit

    def _mode_in_range(self, mode: int) -> bool:
        """
        Check a mode exists within the created circuit and also confirm it is
        an integer.
        """
        if isinstance(mode, Parameter):
            raise TypeError("Mode values cannot be parameters.")
        # Catch this separately as bool is subclass of int
        if isinstance(mode, bool):
            raise TypeError("Mode number should be an integer.")
        if not isinstance(mode, int) and int(mode) != mode:
            raise TypeError("Mode number should be an integer.")
        if not (0 <= mode < self.n_modes):
            raise ModeRangeError(
                "Selected mode(s) is not within the range of the created "
                "circuit."
            )
        return True

    def _map_mode(self, mode: int) -> int:
        """
        Maps a provided mode to the corresponding internal mode
        """
        for i in sorted(self.__internal_modes):
            if mode >= i:
                mode += 1
        return mode

    def _add_empty_mode(
        self, circuit_spec: list[Component], mode: int
    ) -> list[Component]:
        """
        Adds an empty mode at the selected location to a provided circuit spec.
        """
        self.__n_modes += 1
        new_circuit_spec = add_empty_mode_to_circuit_spec(circuit_spec, mode)
        # Also modify heralds as required
        to_modify = [
            "__in_heralds",
            "__out_heralds",
            "__external_in_heralds",
            "__external_out_heralds",
        ]
        for tm in to_modify:
            new_heralds = {}
            for m, n in getattr(self, "_PhotonicCircuit" + tm).items():
                m += 1 if m >= mode else 0  # noqa: PLW2901
                new_heralds[m] = n
            setattr(self, "_PhotonicCircuit" + tm, new_heralds)
        # Add internal mode storage
        self.__internal_modes = [
            m + 1 if m >= mode else m for m in self.__internal_modes
        ]
        return new_circuit_spec

    def _freeze_params(self, circuit_spec: list[Component]) -> list[Component]:
        """
        Takes a provided circuit spec and will remove get any Parameter objects
        with their currently set values.
        """
        new_spec: list[Component] = []
        # Loop over spec and either call function again or add the value to the
        # new spec
        for spec in circuit_spec:
            spec = copy(spec)  # noqa: PLW2901
            if isinstance(spec, Group):
                spec.circuit_spec = self._freeze_params(spec.circuit_spec)
                new_spec.append(spec)
            else:
                for name, value in zip(
                    spec.fields(), spec.values(), strict=True
                ):
                    if isinstance(value, Parameter):
                        setattr(spec, name, value.get())
                new_spec.append(spec)
        return new_spec

    def _get_circuit_spec(self) -> list[Component]:
        """Returns a copy of the circuit spec attribute."""
        return deepcopy(self.__circuit_spec)
