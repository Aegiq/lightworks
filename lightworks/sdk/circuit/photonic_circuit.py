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

from collections.abc import Iterable
from copy import copy, deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
from IPython import display as ipy_display
from numpy.typing import NDArray

from lightworks.sdk.utils.exceptions import (
    CircuitCompilationError,
    LightworksDependencyError,
    ModeRangeError,
)
from lightworks.sdk.utils.param_unitary import ParameterizedUnitary
from lightworks.sdk.visualisation import display

from .parameters import Parameter
from .photonic_circuit_utils import (
    add_empty_mode_to_circuit_spec,
    add_modes_to_circuit_spec,
    check_loss,
    compress_mode_swaps,
    convert_non_adj_beamsplitters,
    find_optimal_mode_swapping,
    unpack_circuit_spec,
)
from .photonic_compiler import CompiledPhotonicCircuit
from .photonic_components import (
    Barrier,
    BeamSplitter,
    Component,
    Group,
    HeraldData,
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
        if self.heralds.input or value.heralds.input:
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
        return self.n_modes - len(self.heralds.input)

    @property
    def heralds(self) -> HeraldData:
        """
        A dictionary which details the set heralds on the inputs and outputs of
        the circuit.
        """
        return HeraldData(
            input=copy(self.__in_heralds), output=copy(self.__out_heralds)
        )

    @property
    def _internal_modes(self) -> list[int]:
        return self.__internal_modes

    @property
    def _external_heralds(self) -> HeraldData:
        """
        Stores details of heralds which are on the outside of a circuit (i.e.
        were not added as part of a group).
        """
        return HeraldData(
            input=copy(self.__external_in_heralds),
            output=copy(self.__external_out_heralds),
        )

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
        circuit = circuit.copy()
        # Force grouping if heralding included
        group = True if circuit.heralds.input else group
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
            circuit.unpack_groups()
        # Check circuit size is valid
        n_heralds = len(circuit.heralds.input)
        if mode + circuit.n_modes - n_heralds > self.n_modes:
            raise ModeRangeError("Circuit to add is outside of mode range")

        # Include any existing internal modes into the circuit to be added
        for i in sorted(self.__internal_modes):
            # Need to account for shifts when adding new heralds
            target_mode = i - mode
            for m in circuit.heralds.input:
                if target_mode > m:
                    target_mode += 1
            if 0 <= target_mode < circuit.n_modes:
                circuit._add_empty_mode(target_mode)

        # Then add new modes for heralds from circuit and also add swaps to
        # enforce that the input and output herald are on the same mode
        provisional_swaps = {}
        circuit_heralds = circuit.heralds
        in_herald_modes = list(circuit_heralds.input.keys())
        out_herald_modes = list(circuit_heralds.output.keys())
        for m in sorted(in_herald_modes):
            self._add_empty_mode(mode + m)
            self.__internal_modes.append(mode + m)
            # Current limitation is that heralding should be on the same mode
            # when adding, so use a mode swap to compensate for this.
            herald_loc = in_herald_modes.index(m)
            out_herald = out_herald_modes[herald_loc]
            provisional_swaps[out_herald] = m

        # Convert provisional swaps into full list and add to circuit
        swaps = find_optimal_mode_swapping(provisional_swaps, circuit.n_modes)
        spec = circuit.__circuit_spec
        # Skip for cases where swaps do not alter mode structure
        if list(swaps.keys()) != list(swaps.values()):
            spec.append(ModeSwaps(swaps))
        # Update heralds to enforce input and output are on the same mode
        new_heralds = HeraldData(
            input=circuit.heralds.input,
            output={swaps[m]: n for m, n in circuit.heralds.output.items()},
        )
        # Also add all included heralds to the heralds dict
        for m in new_heralds.input:
            self.__in_heralds[m + mode] = new_heralds.input[m]
            self.__out_heralds[m + mode] = new_heralds.output[m]
        # And shift all components in circuit by required amount
        add_cs = add_modes_to_circuit_spec(spec, mode)

        # Then add circuit spec, adjusting how this is included
        if not group:
            self.__circuit_spec += add_cs
        else:
            self.__circuit_spec.append(
                Group(
                    add_cs, name, mode, mode + circuit.n_modes - 1, new_heralds
                )
            )

    def bs(
        self,
        mode_1: int,
        mode_2: int | None = None,
        reflectivity: float | Parameter[float] = 0.5,
        loss: float | Parameter[float] = 0,
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

            reflectivity (float | Parameter, optional) : The reflectivity value
                of the beam splitter. Defaults to 0.5.

            loss (float | Parameter, optional) : The loss of the beam splitter,
                this should be provided as a decimal value in the range [0,1].

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

    def ps(
        self,
        mode: int,
        phi: float | Parameter[float],
        loss: float | Parameter[float] = 0,
    ) -> None:
        """
        Applies a phase shift to a given mode in the circuit.

        Args:

            mode (int) : The mode on which the phase shift acts.

            phi (float | Parameter) : The angular phase shift to apply.

            loss (float | Parameter, optional) : The loss of the phase shifter,
                this should be provided as a decimal value in the range [0,1].

        """
        mode = self._map_mode(mode)
        self._mode_in_range(mode)
        check_loss(loss)
        self.__circuit_spec.append(PhaseShifter(mode, phi))
        if isinstance(loss, Parameter) or loss > 0:
            self.loss(mode, loss)

    def loss(self, mode: int, loss: float | Parameter[float] = 0) -> None:
        """
        Adds a loss channel to the specified mode of the circuit.

        Args:

            mode (int) : The mode on which the loss channel acts.

            loss (float | Parameter, optional) : The loss applied to the
                selected mode, this should be provided as a decimal value in the
                range [0,1].

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
        self, modes: int | tuple[int, int], photons: int | tuple[int, int]
    ) -> None:
        """
        Add a herald across a selected input/output of the circuit. If only one
        mode is specified then this will be used for both the input and output.

        Args:

            modes (int | tuple) : The mode(s) to use for heralding. If supplied
                as a single value then the same mode will be used for both input
                and output, otherwise a pair of values can be specified to use
                a different input and output.

            photons (int | tuple) : The photon(s) to use for heralding. If
                supplied as a single value then the same number of photons will
                be heralded on the input and output, otherwise a pair of values
                can be specified to use a different number of photons.

        """
        if isinstance(photons, Iterable) and not isinstance(photons, str):
            in_photon, out_photon = photons
        else:
            in_photon = out_photon = photons
        for n in [in_photon, out_photon]:
            if not isinstance(n, int) or isinstance(n, bool):
                raise TypeError(
                    "Number of photons for herald should be a tuple of two "
                    "integers."
                )
        if isinstance(modes, Iterable) and not isinstance(photons, str):
            input_mode, output_mode = modes
        else:
            input_mode = output_mode = modes
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
        self.__in_heralds[input_mode] = in_photon
        self.__out_heralds[output_mode] = out_photon
        self.__external_in_heralds[input_mode] = in_photon
        self.__external_out_heralds[output_mode] = out_photon

    def display(
        self,
        show_parameter_values: bool = False,
        display_loss: bool = False,
        mode_labels: list[str] | None = None,
        display_type: Literal["svg", "mpl"] = "svg",
        display_barriers: bool = False,
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

            display_barriers (bool, optional) : Shows included barriers within
                the created visualization if this is required.

        """
        return_ = display(
            self,
            display_loss=display_loss,
            mode_labels=mode_labels,
            display_type=display_type,
            show_parameter_values=show_parameter_values,
            display_barriers=display_barriers,
        )
        if display_type == "mpl":
            plt.show()
        elif display_type == "svg":
            ipy_display.display(return_)

    def save_figure(
        self,
        path: str | Path,
        svg: bool = True,
        show_parameter_values: bool = False,
        display_loss: bool = False,
        mode_labels: list[str] | None = None,
        display_barriers: bool = False,
    ) -> None:
        """
        Creates a figure of the current circuit and saves this to the provided
        file name.

        Args:

            path (str|Path) : The path to save the figure to. This can be just a
                filename or can also include a directory which the circuit
                should be placed within.

            svg (bool, optional) : Controls whether the figure is saved as an
                svg (True) or png (False). Defaults to svg.

            show_parameter_values (bool, optional) : Shows the values of
                parameters instead of the associated labels if specified.

            display_loss (bool, optional) : Choose whether to display loss
                components in the figure, defaults to False.

            mode_labels (list|None, optional) : Optionally provided a list of
                mode labels which will be used to name the mode something other
                than numerical values. Can be set to None to use default
                values.

            display_barriers (bool, optional) : Shows included barriers within
                the created visualization if this is required.

        """
        figure = display(
            self,
            display_loss=display_loss,
            mode_labels=mode_labels,
            display_type="svg",
            show_parameter_values=show_parameter_values,
            display_barriers=display_barriers,
        )
        p = Path(path)
        p = p.with_suffix(".svg") if svg else p.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        if svg:
            figure.save_svg(p)
        else:
            figure.set_pixel_scale(5)
            try:
                figure.save_png(str(p))
            except ImportError as e:
                raise LightworksDependencyError(
                    "Unable to save figure as png due to missing dependency."
                    "Please see original exception for more information or "
                    "seek support. Alternatively, it will still be possible to "
                    "save the figure as an svg."
                ) from e

    def get_all_params(self) -> list[Parameter[Any]]:
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
        for i, o in zip(heralds.input, heralds.output, strict=True):
            circuit.add_herald(i, o, heralds.input[i], heralds.output[o])

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

    def _add_empty_mode(self, mode: int) -> None:
        """
        Adds an empty mode at the selected location to a provided circuit spec.
        """
        self.__n_modes += 1
        self.__circuit_spec = add_empty_mode_to_circuit_spec(
            self.__circuit_spec, mode
        )
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
                    if isinstance(value, ParameterizedUnitary):
                        setattr(spec, name, value.unitary)
                new_spec.append(spec)
        return new_spec

    def _get_circuit_spec(self) -> list[Component]:
        """Returns a copy of the circuit spec attribute."""
        return deepcopy(self.__circuit_spec)
