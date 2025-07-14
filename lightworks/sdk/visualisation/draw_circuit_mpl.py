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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import matplotlib.figure
import matplotlib.pyplot as plt
from multimethod import multimethod

from lightworks.sdk.circuit.photonic_components import (
    Barrier,
    BeamSplitter,
    Group,
    HeraldData,
    Loss,
    ModeSwaps,
    PhaseShifter,
    UnitaryMatrix,
)
from lightworks.sdk.utils.exceptions import DisplayError

from .display_utils import MPLSettings, process_parameter_value
from .draw_specs import (
    BarrierDrawing,
    BeamSplitterDrawing,
    HeraldDrawing,
    LossDrawing,
    ModeSwapDrawing,
    PhaseShifterDrawing,
    UnitaryDrawing,
    WaveguideDrawing,
)

if TYPE_CHECKING:
    from lightworks.sdk.circuit import PhotonicCircuit


class DrawCircuitMPL:
    """
    DrawCircuit

    This class can be used to Display a circuit in the quantum emulator as a
    figure in matplotlib.

    Args:

        circuit (PhotonicCircuit) : The circuit which is to be displayed.

        display_loss (bool, optional) : Choose whether to display loss
            components in the figure, defaults to False.

        mode_label (list|None, optional) : Optionally provided a list of mode
            labels which will be used to name the mode something other than
            numerical values. Can be set to None to use default values.

        show_parameter_values (bool, optional) : Shows the values of parameters
            instead of the associated labels if specified.

        display_barriers (bool, optional) : Shows included barriers within the
            created visualization if this is required.

    """

    def __init__(
        self,
        circuit: "PhotonicCircuit",
        display_loss: bool = False,
        mode_labels: list[str] | None = None,
        show_parameter_values: bool = False,
        display_barriers: bool = False,
    ) -> None:
        self.circuit = circuit
        self.display_loss = display_loss
        self.mode_labels = mode_labels
        self.show_parameter_values = show_parameter_values
        self.n_modes = self.circuit.n_modes
        self.herald_modes = self.circuit._internal_modes
        self.display_barriers = display_barriers

    def draw(self) -> tuple[matplotlib.figure.Figure, plt.Axes]:
        """
        Creates matplotlib figure using the provided circuit spec.
        """
        # Set a waveguide width and get mode number
        self.wg_width = 0.05
        # Adjust size of figure according to circuit with min size 4 and max 40
        s = min(len(self.circuit._get_circuit_spec()), 40)
        s = max(s, 4)
        # Create fig and set aspect to equal
        self.fig, self.ax = plt.subplots(figsize=(s, s), dpi=200)
        self.ax.set_aspect("equal")
        # Manually adjust figure height
        h = max(self.n_modes, 4)
        self.fig.set_figheight(h)
        dy = 1.0
        self.dy_smaller = 0.6
        self.y_locations = []
        # Set mode y locations
        yloc = 0.0
        for i in range(self.n_modes):
            self.y_locations.append(yloc)
            if i + 1 in self.herald_modes or i in self.herald_modes:
                yloc += self.dy_smaller
            else:
                yloc += dy
        # Set a starting length and add a waveguide for each mode
        init_length = 0.5
        if False:
            self._add_wg(0, i - self.wg_width / 2, init_length)
        # Create a list to store the positions of each mode
        self.x_locations = [init_length] * self.n_modes
        # Add extra waveguides when using heralds
        if self.circuit._external_heralds.input:
            for i in range(self.n_modes):
                if i not in self.herald_modes:
                    self._add_wg(self.x_locations[i], self.y_locations[i], 0.5)
                    self.x_locations[i] += 0.5
        # Loop over circuit spec and add each component
        for spec in self.circuit._get_circuit_spec():
            self._add(spec)

        # Add any final lengths as required
        final_loc = max(self.x_locations)
        # Extend final waveguide if herald included
        if self.circuit._external_heralds.output:
            final_loc += 0.5
        for i, loc in enumerate(self.x_locations):
            if loc < final_loc and i not in self.herald_modes:
                length = final_loc - loc
                self._add_wg(loc, self.y_locations[i], length)
                self.x_locations[i] += length

        # Add heralding display
        self._add_heralds(
            self.circuit._external_heralds, init_length, final_loc
        )

        # Set axes limits using locations and mode numbers
        self.ax.set_xlim(0, max(self.x_locations) + 0.5)
        self.ax.set_ylim(max(self.y_locations) + 1, -1)
        self.ax.set_yticks(self.y_locations)
        if self.mode_labels is not None:
            exp_len = self.n_modes - len(self.herald_modes)
            if len(self.mode_labels) != exp_len:
                msg = (
                    "Length of provided mode labels list should be equal to "
                    f"the number of useable modes ({exp_len})."
                )
                raise DisplayError(msg)
            mode_labels = self.mode_labels
        else:
            mode_labels = [
                str(i) for i in range(self.n_modes - len(self.herald_modes))
            ]
        mode_labels = [str(m) for m in mode_labels]
        # Include heralded modes in mode labels
        full_mode_labels = []
        count = 0
        for i in range(self.n_modes):
            if i not in self.herald_modes:
                full_mode_labels.append(mode_labels[count])
                count += 1
            else:
                full_mode_labels.append("-")
        self.ax.set_yticklabels(full_mode_labels)
        self.ax.set_xticks([])

        return self.fig, self.ax

    def _add_wg(self, x: float, y: float, length: float) -> None:
        """
        Add a waveguide to the axis.
        """
        WaveguideDrawing(x=x, y=y, length=length, width=self.wg_width).draw_mpl(
            self.ax
        )

    @multimethod
    def _add(self, spec: Any) -> None:  # noqa: ARG002
        """
        Catch all for any components which may not have been implemented.
        """
        raise ValueError("Unrecognised component in circuit spec.")

    @_add.register
    def _add_ps(self, spec: PhaseShifter) -> None:
        """
        Add a phase shifter to the axis.
        """
        phi = process_parameter_value(spec.phi, self.show_parameter_values)
        # Set size of phase shifter box and length of connector
        size = 0.5
        con_length = 0.5
        # Get x and y locs of target modes
        xloc = self.x_locations[spec.mode]
        yloc = self.y_locations[spec.mode]
        # Add input waveguides
        self._add_wg(xloc, yloc, con_length)
        xloc += con_length
        # Add phase shifter square
        PhaseShifterDrawing(x=xloc, y=yloc, size=size, phase=phi).draw_mpl(
            self.ax
        )
        xloc += size
        # Add output waveguides
        self._add_wg(xloc, yloc, con_length)
        # Update mode locations list
        self.x_locations[spec.mode] = xloc + con_length

    @_add.register
    def _add_bs(self, spec: BeamSplitter) -> None:
        """
        Add a beam splitter across to provided modes to the axis.
        """
        mode1, mode2 = spec.mode_1, spec.mode_2
        ref = process_parameter_value(
            spec.reflectivity, self.show_parameter_values
        )
        if mode1 > mode2:
            mode1, mode2 = mode2, mode1
        size_x = 0.5  # x beam splitter size
        con_length = 0.5  # input/output waveguide length
        offset = 0.5  # Offset of beam splitter shape from mode centres
        size_y = offset + abs(
            self.y_locations[mode2] - self.y_locations[mode1]
        )  # Find y size
        # Get x and y locations
        yloc = self.y_locations[min(mode1, mode2)]
        # Equalise all x waveguides
        xloc = self._equalize_waveguides(range(mode1, mode2 + 1))
        # Add input waveguides for all included modes
        modes = range(min(mode1, mode2), max(mode1, mode2) + 1, 1)
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
        xloc += con_length
        # Add beam splitter rectangle shape
        BeamSplitterDrawing(
            x=xloc,
            y=yloc,
            size_x=size_x,
            size_y=size_y,
            offset_y=offset,
            reflectivity=ref,
            text_offset=0,
        ).draw_mpl(self.ax)
        # For any modes in between the beam splitter modes add a waveguide
        # across the beam splitter
        for i in range(mode1 + 1, mode2):
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], size_x)
        xloc += size_x
        # Add output waveguides and update mode locations
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
            self.x_locations[i] = xloc + con_length

    @_add.register
    def _add_loss(self, spec: Loss) -> None:
        """
        Add a loss channel to the specified mode.
        """
        # Don't add if not enabled
        if not self.display_loss:
            return
        loss = process_parameter_value(spec.loss, self.show_parameter_values)
        # Set size of loss element and input/output waveguide length
        size = 0.5
        con_length = 0.5
        # Get x and y locations
        xloc = self.x_locations[spec.mode]
        yloc = self.y_locations[spec.mode]
        # Add an input waveguide
        self._add_wg(xloc, yloc, con_length)
        xloc += con_length
        # Add loss elements
        LossDrawing(x=xloc, y=yloc, size=size, loss=loss).draw_mpl(self.ax)
        xloc += size
        # Add output waveguide
        self._add_wg(xloc, yloc, con_length)
        # Update mode position
        self.x_locations[spec.mode] = xloc + con_length

        return

    @_add.register
    def _add_barrier(self, spec: Barrier) -> None:
        """
        Add a barrier which will separate different parts of the circuit. This
        is applied to the provided modes.
        """
        max_loc = 0.0
        for m in spec.modes:
            max_loc = max(max_loc, self.x_locations[m])
        for m in spec.modes:
            loc = self.x_locations[m]
            if loc < max_loc:
                self._add_wg(loc, self.y_locations[m], max_loc - loc)
            self.x_locations[m] = max_loc
            if self.display_barriers:
                BarrierDrawing(
                    x=max_loc,
                    y_start=self.y_locations[m] - self.dy_smaller / 2,
                    y_end=self.y_locations[m] + self.dy_smaller / 2,
                    width=self.wg_width * 0.6,
                ).draw_mpl(self.ax)

    @_add.register
    def _add_mode_swaps(self, spec: ModeSwaps) -> None:
        """
        Add mode swaps between provided modes to the axis.
        """
        # Skip any empty elements
        if not spec.swaps:
            return
        swaps = dict(spec.swaps)  # Make copy of swaps
        size_x = 1  # x beam splitter size
        con_length = 0.25  # input/output waveguide length
        min_mode = min(swaps)
        max_mode = max(swaps)
        # Add in missing mode for swap
        for m in range(min_mode, max_mode + 1):
            if m not in swaps:
                swaps[m] = m
        # Get y locations
        ylocs = []
        for i, j in swaps.items():
            if i not in self.herald_modes:
                ylocs.append((self.y_locations[i], self.y_locations[j]))
        # Equalise all x waveguides
        xloc = self._equalize_waveguides(range(min_mode, max_mode + 1))
        # Add input waveguides for all included modes
        modes = range(min_mode, max_mode + 1, 1)
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
        xloc += con_length
        ModeSwapDrawing(
            x=xloc, ys=ylocs, size_x=size_x, wg_width=self.wg_width
        ).draw_mpl(self.ax)
        xloc += size_x
        # Add output waveguides update mode locations
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
            self.x_locations[i] = xloc + con_length

        return

    @_add.register
    def _add_unitary(self, spec: UnitaryMatrix) -> None:
        """
        Add a unitary representation to the axis.
        """
        mode1, mode2 = spec.mode, spec.mode + spec.unitary.shape[0] - 1
        size_x = 1  # Unitary x size
        con_length = 0.5  # Input/output waveguide lengths
        offset = 0.5  # Offset of unitary square from modes
        size_y = offset + abs(
            self.y_locations[mode2] - self.y_locations[mode1]
        )  # Find total unitary size
        # Get x and y positions
        yloc = self.y_locations[min(mode1, mode2)]
        # Equalise all x waveguides
        xloc = self._equalize_waveguides(range(mode1, mode2 + 1))
        # Add input waveguides
        modes = range(min(mode1, mode2), max(mode1, mode2) + 1, 1)
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
        xloc += con_length
        # Add unitary shape and label
        UnitaryDrawing(
            x=xloc,
            y=yloc,
            size_x=size_x,
            size_y=size_y,
            offset_y=offset,
            label=spec.label,
            text_size=MPLSettings.TEXT_SIZE.value,
        ).draw_mpl(self.ax)
        xloc += size_x
        # Add output waveguides and update mode positions
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
            self.x_locations[i] = xloc + con_length

    @_add.register
    def _add_grouped_circuit(self, spec: Group) -> None:
        """
        Add a grouped circuit drawing to the axis.
        """
        mode1, mode2 = spec.mode_1, spec.mode_2
        if mode1 > mode2:
            mode1, mode2 = mode2, mode1
        size_x = 1  # x size
        con_length = 0.5  # Input/output waveguide lengths
        extra_length = 0.5 if spec.heralds.input or spec.heralds.output else 0
        offset = 0.5  # Offset of square from modes
        size_y = offset + abs(
            self.y_locations[mode2] - self.y_locations[mode1]
        )  # Find total unitary size
        # Get x and y positions
        yloc = self.y_locations[min(mode1, mode2)]
        # Equalise all x waveguides
        xloc = self._equalize_waveguides(range(mode1, mode2 + 1))
        # Add input waveguides
        modes = range(min(mode1, mode2), max(mode1, mode2) + 1, 1)
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(
                    xloc, self.y_locations[i], con_length + extra_length
                )
            elif i - mode1 in spec.heralds.input:
                self._add_wg(
                    xloc + extra_length, self.y_locations[i], con_length
                )
        xloc += con_length + extra_length
        # Add circuit shape and label
        UnitaryDrawing(
            x=xloc,
            y=yloc,
            size_x=size_x,
            size_y=size_y,
            offset_y=offset,
            label=spec.name,
            text_size=MPLSettings.TEXT_SIZE.value,
        ).draw_mpl(self.ax)
        xloc += size_x
        # Add output waveguides and update mode positions
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(
                    xloc, self.y_locations[i], con_length + extra_length
                )
            elif i - mode1 in spec.heralds.output:
                self._add_wg(xloc, self.y_locations[i], con_length)
            self.x_locations[i] = xloc + con_length + extra_length

        # Modify provided heralds by mode offset and then add at locations
        shifted_heralds = HeraldData(
            input={m + mode1: n for m, n in spec.heralds.input.items()},
            output={m + mode1: n for m, n in spec.heralds.output.items()},
        )
        self._add_heralds(
            shifted_heralds, xloc - size_x - con_length, xloc + con_length
        )

    def _add_heralds(
        self,
        heralds: HeraldData,
        start_loc: float,
        end_loc: float,
    ) -> None:
        """
        Adds display of all heralds to circuit.
        """
        size = 0.2
        # Input heralds
        for mode, num in heralds.input.items():
            xloc = start_loc
            yloc = self.y_locations[mode]
            HeraldDrawing(x=xloc, y=yloc, size=size, n_photons=num).draw_mpl(
                self.ax
            )
        # Output heralds
        for mode, num in heralds.output.items():
            xloc = end_loc
            yloc = self.y_locations[mode]
            HeraldDrawing(x=xloc, y=yloc, size=size, n_photons=num).draw_mpl(
                self.ax
            )

    def _equalize_waveguides(self, modes: Iterable[int]) -> float:
        """
        Extends the waveguide of any modes in the list to the target position.
        """
        all_locations = [self.x_locations[m] for m in modes]
        xloc = max(all_locations)
        for mode, loc in zip(modes, all_locations, strict=True):
            if loc < xloc and mode not in self.herald_modes:
                self._add_wg(loc, self.y_locations[mode], xloc - loc)
        return xloc
