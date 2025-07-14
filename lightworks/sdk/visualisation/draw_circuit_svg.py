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

import drawsvg as draw
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

from .display_utils import SVGSettings, process_parameter_value
from .draw_specs import (
    BarrierDrawing,
    BeamSplitterDrawing,
    DrawSpec,
    HeraldDrawing,
    LossDrawing,
    ModeSwapDrawing,
    PhaseShifterDrawing,
    TextDrawing,
    UnitaryDrawing,
    WaveguideDrawing,
)

if TYPE_CHECKING:
    from lightworks.sdk.circuit import PhotonicCircuit


class DrawCircuitSVG:
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
        self.herald_modes = self.circuit._internal_modes
        self.show_parameter_values = show_parameter_values
        self.display_barriers = display_barriers
        # Set a waveguide width and get mode number
        self.wg_width = 8
        self.n_modes = self.circuit.n_modes
        self.draw_spec: list[DrawSpec] = []
        self.dy = 125
        self.dy_smaller = 75
        self.y_locations = []
        # Set mode y locations
        yloc = self.dy
        for i in range(self.n_modes):
            self.y_locations.append(yloc)
            if i + 1 in self.herald_modes or i in self.herald_modes:
                yloc += self.dy_smaller
            else:
                yloc += self.dy
        init_length = 100.0
        # Add the mode labels
        if mode_labels is not None:
            exp_len = self.n_modes - len(self.herald_modes)
            if len(mode_labels) != exp_len:
                msg = (
                    "Length of provided mode labels list should be equal to "
                    f"the number of useable modes ({exp_len})."
                )
                raise DisplayError(msg)
        else:
            mode_labels = [
                str(i) for i in range(self.n_modes - len(self.herald_modes))
            ]
        # Convert all labels to strings
        mode_labels = [str(m) for m in mode_labels]
        full_mode_labels = []
        count = 0
        for i in range(self.n_modes):
            if i not in self.herald_modes:
                full_mode_labels.append(mode_labels[count])
                count += 1
            else:
                full_mode_labels.append("-")
        mode_labels = full_mode_labels
        # Adjust canvas size for long labels
        max_len = max(len(m) for m in mode_labels)
        if max_len > 4:
            init_length += (max_len - 4) * 17.5
        for m, label in enumerate(mode_labels):
            self.draw_spec.append(
                TextDrawing(
                    text=label,
                    x=init_length - 20,
                    y=self.y_locations[m] + 2,
                    rotation=0,
                    size=SVGSettings.TEXT_SIZE.value,
                    colour="black",
                    alignment="right",
                )
            )

        self._init_length = init_length
        # Create list of locations for each mode
        self.x_locations = [init_length + 50] * self.n_modes
        # Add extra waveguides when using heralds
        if self.circuit._external_heralds.input:
            for m in range(self.n_modes):
                if m not in self.herald_modes:
                    self._add_wg(self.x_locations[m], self.y_locations[m], 50)
                    self.x_locations[m] += 50
        # Loop over each element in the circuit spec and add
        for spec in self.circuit._get_circuit_spec():
            self._add(spec)

        maxloc = max(self.x_locations)
        # Extend final waveguide if herald included
        if self.circuit._external_heralds.output:
            maxloc += 50
        for i, loc in enumerate(self.x_locations):
            if loc < maxloc and i not in self.herald_modes:
                self._add_wg(loc, self.y_locations[i], maxloc - loc)
                self.x_locations[i] = maxloc

        # Add heralding display
        self._add_heralds(
            self.circuit._external_heralds, self._init_length + 50, maxloc
        )

        for i in range(self.n_modes):
            self.x_locations[i] += 50

    def draw(self) -> draw.Drawing:
        """
        Draws the circuit as defined in the initial class class:

        Returns:

            draw.Drawing : The created drawing with all circuit components.

        """
        extra_lower = 0
        # Create a new drawing of the correct size
        self.d = draw.Drawing(
            max(self.x_locations) + 100,
            max(self.y_locations) + self.dy + extra_lower,
        )

        # Set white background
        self.d.append(
            draw.Rectangle(
                0, 0, self.d.width, self.d.height, fill="white", stroke="none"
            )
        )

        border = 100
        # Add frame around circuit
        dx = max(self.x_locations) - self._init_length
        dy = max(self.y_locations) - self.dy + 2 * border + extra_lower
        self.d.append(
            draw.Rectangle(
                self._init_length,
                self.dy - border,
                dx,
                dy,
                fill="none",
                stroke="black",
            )
        )
        # Add ticks to left side of frame
        length, width = 8, 1
        for i in range(self.n_modes):
            self.d.append(
                draw.Rectangle(
                    self._init_length - length,
                    self.y_locations[i] - width / 2,
                    length,
                    width,
                    fill="black",
                )
            )

        for spec in self.draw_spec:
            self.d.append(spec.draw_svg())

        mode_width = self.circuit.n_modes - 0.6 * len(
            self.circuit._internal_modes
        )
        # Adjust size of figure to meet target scale
        target_scale = min(50 + mode_width * 65, 900)
        self.d.set_pixel_scale(1)
        _w, h = self.d.calc_render_size()
        self.d.set_pixel_scale(target_scale / h)

        return self.d

    def _add_wg(self, x: float, y: float, length: float) -> None:
        """
        Add a waveguide to the drawing.
        """
        # Add a waveguide of the required length to the drawing spec
        self.draw_spec.append(
            WaveguideDrawing(
                x=x - 0.05, y=y, length=length + 0.1, width=self.wg_width
            )
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
        Add a phase shifter to the drawing.
        """
        phi = process_parameter_value(spec.phi, self.show_parameter_values)
        size = 50
        con_length = 50
        # Get current x and y locations for the mode
        xloc = self.x_locations[spec.mode]
        yloc = self.y_locations[spec.mode]
        # Input waveguide
        self._add_wg(xloc, yloc, con_length)
        xloc += con_length
        # Add phase shifter section
        self.draw_spec.append(
            PhaseShifterDrawing(x=xloc, y=yloc, size=size, phase=phi)
        )
        xloc += size
        # Output waveguide
        self._add_wg(xloc, yloc, con_length)
        self.x_locations[spec.mode] = xloc + con_length

    @_add.register
    def _add_bs(self, spec: BeamSplitter) -> None:
        """
        Add a beam splitter across to provided modes to the drawing.
        """
        mode1, mode2 = spec.mode_1, spec.mode_2
        ref = process_parameter_value(
            spec.reflectivity, self.show_parameter_values
        )
        if mode1 > mode2:
            mode1, mode2 = mode2, mode1
        size_x = 50  # x beam splitter size
        con_length = 50  # input/output waveguide length
        offset = 50  # Offset of beam splitter shape from mode centres
        if mode1 > mode2:
            mode1, mode2 = mode2, mode1
        size_y = offset + self.y_locations[mode2] - self.y_locations[mode1]
        # Get x and y locations
        yloc = self.y_locations[mode1]
        # Equalise all x waveguides
        xloc = self._equalize_waveguides(range(mode1, mode2 + 1))
        # Add input waveguides for all included modes
        for i in range(mode1, mode2 + 1):
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
        xloc += con_length
        # Check for any required text offset to avoid waveguides
        mode_between = 0
        for i in range(mode1 + 1, mode2 + 1, 1):
            if i not in self.herald_modes:
                mode_between += 1
                dy = -50 if mode_between % 2 == 0 else 0
        # Add beam splitter section
        self.draw_spec.append(
            BeamSplitterDrawing(
                x=xloc,
                y=yloc,
                size_x=size_x,
                size_y=size_y,
                offset_y=offset / 2,
                reflectivity=ref,
                text_offset=dy,
            )
        )
        # For any modes in between the beam splitter modes add a waveguide
        # across the beam splitter
        for i in range(mode1 + 1, mode2):
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], size_x)
        xloc += size_x
        # Add output waveguides and update mode locations
        for i in range(mode1, mode2 + 1):
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
            self.x_locations[i] = xloc + con_length

    @_add.register
    def _add_unitary(self, spec: UnitaryMatrix) -> None:
        """
        Add a unitary matrix representation to the drawing.
        """
        mode1, mode2 = spec.mode, spec.mode + spec.unitary.shape[0] - 1
        size_x = 100  # Unitary x size
        con_length = 50  # Input/output waveguide lengths
        offset = 50  # Offset of unitary square from modes
        size_y = offset + self.y_locations[mode2] - self.y_locations[mode1]
        # Get x and y positions
        yloc = self.y_locations[mode1]
        # Equalise all x waveguides
        xloc = self._equalize_waveguides(range(mode1, mode2 + 1))
        # Add input waveguides
        modes = range(mode1, mode2 + 1, 1)
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
        xloc += con_length
        # Add unitary shape and U label
        self.draw_spec.append(
            UnitaryDrawing(
                x=xloc,
                y=yloc,
                size_x=size_x,
                size_y=size_y,
                offset_y=offset / 2,
                label=spec.label,
                text_size=SVGSettings.TEXT_SIZE.value
                if self.n_modes > 2
                else SVGSettings.U_TEXT_SIZE.value,
            )
        )
        xloc += size_x
        # Add output waveguides and update mode positions
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
        size = 50
        con_length = 50
        # Get current x and y locations for the mode
        xloc = self.x_locations[spec.mode]
        yloc = self.y_locations[spec.mode]
        # Input waveguide
        self._add_wg(xloc, yloc, con_length)
        xloc += con_length
        # Add phase shifter section
        self.draw_spec.append(LossDrawing(x=xloc, y=yloc, size=size, loss=loss))
        xloc += size
        # Output waveguide
        self._add_wg(xloc, yloc, con_length)
        self.x_locations[spec.mode] = xloc + con_length

        return

    @_add.register
    def _add_barrier(self, spec: Barrier) -> None:
        """
        Add a barrier which will separate different parts of the circuit. This
        is applied to the provided modes.
        """
        max_loc = max(self.x_locations[m] for m in spec.modes)
        for m in spec.modes:
            loc = self.x_locations[m]
            if loc < max_loc:
                self._add_wg(loc, self.y_locations[m], max_loc - loc)
            self.x_locations[m] = max_loc
            if self.display_barriers:
                self.draw_spec.append(
                    BarrierDrawing(
                        x=max_loc,
                        y_start=self.y_locations[m] - self.dy_smaller / 2,
                        y_end=self.y_locations[m] + self.dy_smaller / 2,
                        width=self.wg_width * 0.6,
                    )
                )

    @_add.register
    def _add_mode_swaps(self, spec: ModeSwaps) -> None:
        """
        Add mode swaps between provided modes to the drawing.
        """
        # Skip any empty elements
        if not spec.swaps:
            return
        swaps = dict(spec.swaps)  # Make copy of swaps
        con_length = 25  # input/output waveguide length
        min_mode = min(swaps)
        max_mode = max(swaps)
        size_x = 50 + 20 * (max_mode - min_mode)  # x length of swap element
        # Add in missing mode for swap
        for m in range(min_mode, max_mode + 1):
            if m not in swaps:
                swaps[m] = m
        # Get y locations
        ylocs = []
        for i, j in swaps.items():
            if i not in self.herald_modes:
                ylocs.append(
                    (float(self.y_locations[i]), float(self.y_locations[j]))
                )
        # Equalise all x waveguides
        xloc = self._equalize_waveguides(range(min_mode, max_mode + 1))
        # Add input waveguides for all included modes
        modes = range(min_mode, max_mode + 1, 1)
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
        xloc += con_length
        # Add beam splitter section
        self.draw_spec.append(
            ModeSwapDrawing(
                x=xloc, ys=ylocs, size_x=size_x, wg_width=self.wg_width
            )
        )
        xloc += size_x
        # Add output waveguides update mode locations
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
            self.x_locations[i] = xloc + con_length

        return

    @_add.register
    def _add_grouped_circuit(self, spec: Group) -> None:
        """
        Add a grouped_circuit representation to the drawing.
        """
        mode1, mode2 = spec.mode_1, spec.mode_2
        if mode1 > mode2:
            mode1, mode2 = mode2, mode1
        size_x = 100  # Drawing x size
        con_length = 50  # Input/output waveguide lengths
        extra_length = 50 if spec.heralds.input or spec.heralds.output else 0
        offset = 50  # Offset of square from modes
        size_y = offset + self.y_locations[mode2] - self.y_locations[mode1]
        # Get x and y positions
        yloc = self.y_locations[mode1]
        # Equalise all x waveguides
        xloc = self._equalize_waveguides(range(mode1, mode2 + 1))
        # Add input waveguides
        modes = range(mode1, mode2 + 1, 1)
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
        # Add unitary shape and U label
        self.draw_spec.append(
            UnitaryDrawing(
                x=xloc,
                y=yloc,
                size_x=size_x,
                size_y=size_y,
                offset_y=offset / 2,
                label=spec.name,
                text_size=SVGSettings.TEXT_SIZE.value
                if self.n_modes > 2
                else SVGSettings.U_TEXT_SIZE.value,
            )
        )
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
        size = 25
        # Input heralds
        for mode, num in heralds.input.items():
            xloc = start_loc
            yloc = self.y_locations[mode]
            self.draw_spec.append(
                HeraldDrawing(x=xloc, y=yloc, size=size, n_photons=num)
            )
        # Output heralds
        for mode, num in heralds.output.items():
            xloc = end_loc
            yloc = self.y_locations[mode]
            self.draw_spec.append(
                HeraldDrawing(x=xloc, y=yloc, size=size, n_photons=num)
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
