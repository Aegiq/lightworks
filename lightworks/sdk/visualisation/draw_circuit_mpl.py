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

from typing import TYPE_CHECKING, Any

import matplotlib.figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from multimethod import multimethod

from ..circuit.photonic_components import (
    Barrier,
    BeamSplitter,
    Group,
    Loss,
    ModeSwaps,
    PhaseShifter,
    UnitaryMatrix,
)
from ..utils import DisplayError
from .display_utils import process_parameter_value

if TYPE_CHECKING:
    from ..circuit import PhotonicCircuit


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

    """

    def __init__(
        self,
        circuit: "PhotonicCircuit",
        display_loss: bool = False,
        mode_labels: list[str] | None = None,
        show_parameter_values: bool = False,
    ) -> None:
        self.circuit = circuit
        self.display_loss = display_loss
        self.mode_labels = mode_labels
        self.show_parameter_values = show_parameter_values
        self.n_modes = self.circuit.n_modes
        self.herald_modes = self.circuit._internal_modes

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
        dy_smaller = 0.6
        self.y_locations = []
        # Set mode y locations
        yloc = 0.0
        for i in range(self.n_modes):
            self.y_locations.append(yloc)
            if i + 1 in self.herald_modes or i in self.herald_modes:
                yloc += dy_smaller
            else:
                yloc += dy
        # Set a starting length and add a waveguide for each mode
        init_length = 0.5
        if False:
            self._add_wg(0, i - self.wg_width / 2, init_length)
        # Create a list to store the positions of each mode
        self.x_locations = [init_length] * self.n_modes
        # Add extra waveguides when using heralds
        if self.circuit._external_heralds["input"]:
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
        if self.circuit._external_heralds["output"]:
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
        # Create rectangle using provided length and waveguide width and then
        # add patch
        rect = patches.Rectangle(
            (x, y - self.wg_width / 2), length, self.wg_width, facecolor="black"
        )
        self.ax.add_patch(rect)

        return

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
        rect = patches.Rectangle(
            (xloc, yloc - size / 2),
            size,
            size,
            facecolor="#e8532b",
            edgecolor="black",
        )
        self.ax.add_patch(rect)
        # Include text to label applied phase shift
        self.ax.text(
            xloc + size / 2,
            yloc,
            "PS",
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            size=8,
        )
        # Work out value of n*pi/4 closest to phi
        if not isinstance(phi, str):
            n = int(np.round(phi / (np.pi / 4)))
            # Check if value of phi == n*pi/4 to 8 decimal places
            if round(phi, 8) == round(n * np.pi / 4, 8):  # and n > 0:
                n = abs(n)
                # Set text with either pi or pi/2 or pi/4
                if n == 0:
                    phi_text = "0"
                elif n % 4 == 0:
                    phi_text = str(int(n / 4)) + "π" if n > 4 else "π"
                elif n % 4 == 2:
                    phi_text = str(int(n / 2)) + "π/2" if n > 2 else "π/2"
                else:
                    phi_text = str(int(n)) + "π/4" if n > 1 else "π/4"
                if phi < 0:
                    phi_text = "-" + phi_text
            # Otherwise round phi to 4 decimal places
            else:
                phi_text = str(round(phi, 4))
            phi_text = f"$\\phi = {phi_text}$"
        else:
            phi_text = "$\\phi =$ " + phi
        self.ax.text(
            xloc + size / 2,
            yloc + size / 2 + 0.15,
            phi_text,
            color="black",
            size=5,
            horizontalalignment="center",
            verticalalignment="center",
        )
        xloc += size
        # Add output waveguides
        self._add_wg(xloc, yloc, con_length)
        # Update mode locations list
        self.x_locations[spec.mode] = xloc + con_length

        return

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
        xloc = max(self.x_locations[mode1 : mode2 + 1])
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.x_locations[mode1 : mode2 + 1]):
            if loc < xloc and i + mode1 not in self.herald_modes:
                self._add_wg(loc, self.y_locations[i + mode1], xloc - loc)
        # Add input waveguides for all included modes
        modes = range(min(mode1, mode2), max(mode1, mode2) + 1, 1)
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
        xloc += con_length
        # Add beam splitter rectangle shape
        rect = patches.Rectangle(
            (xloc, yloc - offset / 2),
            size_x,
            size_y,
            facecolor="#3e368d",
            alpha=1,
            edgecolor="black",
        )
        self.ax.add_patch(rect)
        # Label beam splitter as BS and display reflectivity
        self.ax.text(
            xloc + size_x / 2,
            yloc + 0.5,
            "BS",
            size=8,
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
        )
        if not isinstance(ref, str):
            ref = round(ref, 4)
        self.ax.text(
            xloc + size_x / 2,
            yloc + size_y - offset / 2 + 0.15,
            f"$r =$ {ref}",
            color="black",
            size=5,
            horizontalalignment="center",
            verticalalignment="center",
        )
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

        return

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
        rect = patches.Rectangle(
            (xloc, yloc - size / 2),
            size,
            size,
            facecolor="grey",
            edgecolor="black",
        )
        self.ax.add_patch(rect)
        # Label element and add text will loss amount in dB
        self.ax.text(
            xloc + size / 2,
            yloc,
            "L",
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            size=8,
        )
        if not isinstance(loss, str):
            loss_text = f"$loss = {round(loss, 4)}$"
        else:
            loss_text = "$loss =$ " + loss
        self.ax.text(
            xloc + size / 2,
            yloc + size / 2 + 0.15,
            loss_text,
            color="black",
            size=5,
            horizontalalignment="center",
            verticalalignment="center",
        )
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

        return

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
        # Get x and y locations
        xloc = max(self.x_locations[min_mode : max_mode + 1])
        ylocs = []
        for i, j in swaps.items():
            if i not in self.herald_modes:
                ylocs.append((self.y_locations[i], self.y_locations[j]))
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.x_locations[min_mode : max_mode + 1]):
            if loc < xloc and i + min_mode not in self.herald_modes:
                self._add_wg(loc, self.y_locations[min_mode + i], xloc - loc)
        # Add input waveguides for all included modes
        modes = range(min_mode, max_mode + 1, 1)
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
        xloc += con_length
        for y0, y1 in ylocs:
            w = self.wg_width / 2
            if y0 < y1:
                dx1 = w * np.arctan(abs(y1 - y0) / size_x)
                dx2 = 0
            else:
                dx1 = 0
                dx2 = w * np.arctan(abs(y1 - y0) / size_x)
            points = [
                (xloc + dx1, y0 - w),
                (xloc, y0 - w),
                (xloc, y0 + w),
                (xloc + dx2, y0 + w),
                (xloc + size_x - dx1, y1 + w),
                (xloc + size_x, y1 + w),
                (xloc + size_x, y1 - w),
                (xloc + size_x - dx2, y1 - w),
            ]
            poly = patches.Polygon(points, facecolor="black")
            self.ax.add_patch(poly)
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
        xloc = max(self.x_locations[mode1 : mode2 + 1])
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.x_locations[mode1 : mode2 + 1]):
            if loc < xloc and i + mode1 not in self.herald_modes:
                self._add_wg(loc, self.y_locations[i + mode1], xloc - loc)
        # Add input waveguides
        modes = range(min(mode1, mode2), max(mode1, mode2) + 1, 1)
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
        xloc += con_length
        # Add unitary shape and label
        rect = patches.Rectangle(
            (xloc, yloc - offset / 2),
            size_x,
            size_y,
            facecolor="#1a0f36",
            edgecolor="black",
        )
        self.ax.add_patch(rect)
        s = 10 if len(spec.label) == 1 else 8
        r = 90 if len(spec.label) > 2 else 0
        self.ax.text(
            xloc + size_x / 2,
            yloc + (size_y - offset) / 2,
            spec.label,
            horizontalalignment="center",
            size=s,
            verticalalignment="center",
            color="white",
            rotation=r,
        )
        xloc += size_x
        # Add output waveguides and update mode positions
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(xloc, self.y_locations[i], con_length)
            self.x_locations[i] = xloc + con_length

        return

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
        extra_length = (
            0.5 if spec.heralds["input"] or spec.heralds["output"] else 0
        )
        offset = 0.5  # Offset of square from modes
        size_y = offset + abs(
            self.y_locations[mode2] - self.y_locations[mode1]
        )  # Find total unitary size
        # Get x and y positions
        yloc = self.y_locations[min(mode1, mode2)]
        xloc = max(self.x_locations[mode1 : mode2 + 1])
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.x_locations[mode1 : mode2 + 1]):
            if loc < xloc and i + mode1 not in self.herald_modes:
                self._add_wg(loc, self.y_locations[i + mode1], xloc - loc)
        # Add input waveguides
        modes = range(min(mode1, mode2), max(mode1, mode2) + 1, 1)
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(
                    xloc, self.y_locations[i], con_length + extra_length
                )
            elif i - mode1 in spec.heralds["input"]:
                self._add_wg(
                    xloc + extra_length, self.y_locations[i], con_length
                )
        xloc += con_length + extra_length
        # Add circuit shape and label
        rect = patches.Rectangle(
            (xloc, yloc - offset / 2),
            size_x,
            size_y,
            facecolor="#1a0f36",
            edgecolor="black",
        )
        self.ax.add_patch(rect)
        s = 10 if len(spec.name) == 1 else 8
        r = 90 if len(spec.name) > 2 else 0
        self.ax.text(
            xloc + size_x / 2,
            yloc + (size_y - offset) / 2,
            spec.name,
            horizontalalignment="center",
            size=s,
            verticalalignment="center",
            color="white",
            rotation=r,
        )
        xloc += size_x
        # Add output waveguides and update mode positions
        for i in modes:
            if i not in self.herald_modes:
                self._add_wg(
                    xloc, self.y_locations[i], con_length + extra_length
                )
            elif i - mode1 in spec.heralds["output"]:
                self._add_wg(xloc, self.y_locations[i], con_length)
            self.x_locations[i] = xloc + con_length + extra_length

        # Modify provided heralds by mode offset and then add at locations
        shifted_heralds = {
            "input": {m + mode1: n for m, n in spec.heralds["input"].items()},
            "output": {m + mode1: n for m, n in spec.heralds["output"].items()},
        }
        self._add_heralds(
            shifted_heralds, xloc - size_x - con_length, xloc + con_length
        )

        return

    def _add_heralds(
        self,
        heralds: dict[str, dict[int, int]],
        start_loc: float,
        end_loc: float,
    ) -> None:
        """
        Adds display of all heralds to circuit.
        """
        size = 0.2
        # Input heralds
        for mode, num in heralds["input"].items():
            xloc = start_loc
            yloc = self.y_locations[mode]
            circle = patches.Circle(
                (xloc, yloc), size, facecolor="#3e368d", edgecolor="black"
            )
            self.ax.add_patch(circle)
            self.ax.text(
                xloc,
                yloc + 0.01,
                str(num),
                color="white",
                size=8,
                horizontalalignment="center",
                verticalalignment="center",
            )
        # Output heralds
        for mode, num in heralds["output"].items():
            xloc = end_loc
            yloc = self.y_locations[mode]
            circle = patches.Circle(
                (xloc, yloc), size, facecolor="#3e368d", edgecolor="black"
            )
            self.ax.add_patch(circle)
            self.ax.text(
                xloc,
                yloc + 0.01,
                str(num),
                color="white",
                size=8,
                horizontalalignment="center",
                verticalalignment="center",
            )
