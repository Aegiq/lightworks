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
        self.ax.set_ylim(N, -1)
        # Set a starting length and add a waveguide for each mode
        init_length = 0.5
        if False:
            self._add_wg(0, i-self.wg_width/2, init_length)
        # Create a list to store the positions of each mode
        self.locations = [init_length]*N
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
                self._add_mode_swaps(modes)
            elif c == "unitary":
                m1, m2 = modes
                if m1 > m2:
                    m1, m2 = m2, m1
                self._add_unitary(m1, m2)
            elif c == "group":
                m1, m2 = modes
                if m1 > m2:
                    m1, m2 = m2, m1
                self._add_grouped_circuit(m1, m2, params)
        # Add any final lengths as required
        final_loc = max(self.locations)
        for i, loc in enumerate(self.locations):    
            if loc < final_loc:
                length = final_loc - loc
                rect = patches.Rectangle((loc, i-self.wg_width/2), length, 
                                         self.wg_width, facecolor = "black")
                self.ax.add_patch(rect)
        # Set axes limits using locations and mode numbers
        self.ax.set_xlim(0, max(self.locations) + 0.5)
        self.ax.set_yticks(range(0, N))
        if self.mode_labels is not None:
            if len(self.mode_labels) != N:
                raise DisplayError(
                    "Length of provided mode labels list should be equal to "
                    "the number of modes.")
            self.ax.set_yticklabels(self.mode_labels)
        self.ax.set_xticks([])

        return self.fig, self.ax
    
    