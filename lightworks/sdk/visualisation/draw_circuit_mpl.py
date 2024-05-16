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

from .display_components_mpl import DisplayComponentsMPL
from ..utils import DisplayError

from typing import Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DrawCircuitMPL(DisplayComponentsMPL):
    """
    DrawCircuit
    
    This class can be used to Display a circuit in the quantum emulator as a
    figure in matplotlib.
    
    Args:
    
        circuit (Circuit) : The circuit which is to be displayed.
        
        display_loss (bool, optional) : Choose whether to display loss
            components in the figure, defaults to False.
                                        
        mode_label (list|None, optional) : Optionally provided a list of mode
            labels which will be used to name the mode something other than 
            numerical values. Can be set to None to use default values.
                                           
    """
    
    def __init__(self, circuit: "Circuit", display_loss: bool = False,         # type:ignore - Hide warning raised by "Circuit"
                 mode_labels: list[str] | None = None) -> None:
        
        self.circuit = circuit
        self.display_loss = display_loss
        self.mode_labels = mode_labels
        self.N = self.circuit.n_modes
        self.herald_modes = self.circuit._internal_modes
        
    def draw(self) -> tuple[plt.figure, plt.axes]:
        
        # Set a waveguide width and get mode number
        self.wg_width = 0.05
        N = self.circuit.n_modes
        # Adjust size of figure according to circuit with min size 4 and max 40
        s = min(len(self.circuit._display_spec), 40)
        s = max(s, 4)
        # Create fig and set aspect to equal
        self.fig, self.ax = plt.subplots(figsize = (s, s), dpi = 200)
        self.ax.set_aspect('equal')
        # Manually adjust figure height
        h = max(N, 4)
        self.fig.set_figheight(h)
        dy = 1
        dy_smaller = 0.6
        self.y_locations = []
        # Set mode y locations
        yloc = 0
        for i in range(self.N):
            self.y_locations.append(yloc)
            if i+1 in self.herald_modes:
                yloc += dy_smaller
            elif i in self.herald_modes:
                yloc += dy_smaller
            else:
                yloc += dy
        # Set a starting length and add a waveguide for each mode
        init_length = 0.5
        if False:
            self._add_wg(0, i-self.wg_width/2, init_length)
        # Create a list to store the positions of each mode
        self.x_locations = [init_length]*N
        # Add extra waveguides when using heralds
        if self.circuit._external_heralds["input"]:
            for i in range(self.N):
                if i not in self.herald_modes:
                    self._add_wg(self.x_locations[i], self.y_locations[i], 0.5)
                    self.x_locations[i] += 0.5
        # Loop over build spec and add each component
        for spec in self.circuit._display_spec:
            c, modes = spec[0:2]
            params = spec[2]
            if c == "PS":
                self._add_ps(modes, params)
            elif c == "BS":
                m1, m2 = modes
                ref = params
                if m1 > m2:
                    m1, m2 = m2, m1
                self._add_bs(m1, m2, ref)
            elif c == "LC" and self.display_loss:
                self._add_loss(modes, params)
            elif c == "barrier":
                self._add_barrier(modes)
            elif c == "mode_swaps":
                if not modes:
                    continue
                self._add_mode_swaps(modes)
            elif c == "unitary":
                m1, m2 = modes
                if m1 > m2:
                    m1, m2 = m2, m1
                self._add_unitary(m1, m2, params)
            elif c == "group":
                m1, m2 = modes
                if m1 > m2:
                    m1, m2 = m2, m1
                name, heralds = params
                self._add_grouped_circuit(m1, m2, name, heralds)
        # Add any final lengths as required
        final_loc = max(self.x_locations)
        # Extend final waveguide if herald included
        if (self.circuit._external_heralds["output"]):
            final_loc += 0.5
        for i, loc in enumerate(self.x_locations):    
            if loc < final_loc and i not in self.herald_modes:
                length = final_loc - loc
                self._add_wg(loc, self.y_locations[i], length)
                self.x_locations[i] += length
                
        # Add heralding display
        self._add_heralds(self.circuit._external_heralds, init_length,
                          final_loc)
                
        # Set axes limits using locations and mode numbers
        self.ax.set_xlim(0, max(self.x_locations) + 0.5)
        self.ax.set_ylim(max(self.y_locations) + 1, -1)
        self.ax.set_yticks(self.y_locations)
        if self.mode_labels is not None:
            exp_len = N - len(self.herald_modes)
            if len(self.mode_labels) != exp_len:
                raise DisplayError(
                    "Length of provided mode labels list should be equal to "
                    f"the number of useable modes ({exp_len}).")
            mode_labels = self.mode_labels
        else:
            mode_labels = range(N - len(self.herald_modes))
        mode_labels = [str(m) for m in mode_labels]
        # Include heralded modes in mode labels
        full_mode_labels = []
        count = 0
        for i in range(N):
            if i not in self.herald_modes:
                full_mode_labels.append(mode_labels[count])
                count += 1
            else:
                full_mode_labels.append("-")
        self.ax.set_yticklabels(full_mode_labels)
        self.ax.set_xticks([])

        return self.fig, self.ax
    
    