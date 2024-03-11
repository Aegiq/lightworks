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

from .display_components_svg import SVGDrawSpec, DisplayComponentsSVG
from ..utils import DisplayError

import drawsvg as draw

class DrawCircuitSVG(SVGDrawSpec, DisplayComponentsSVG):
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
                 mode_labels: list[str] = None) -> None:
        
        self.circuit = circuit
        self.display_loss = display_loss
        self.mode_labels = mode_labels
        # Set a waveguide width and get mode number
        self.wg_width = 8
        self.N = self.circuit.n_modes
        self.draw_spec = []
        self.dy = 125
        init_length = 100        
        # Add the mode labels
        if mode_labels is None:
            mode_labels = range(self.N)
        # Convert all labels to strings
        mode_labels = [str(m) for m in mode_labels]
        # Adjust canvas size for long labels
        max_len = max([len(m) for m in mode_labels])
        if max_len > 4: init_length += (max_len-4)*17.5
        for m, label in enumerate(mode_labels):
            self.draw_spec += [("text", (label, init_length-20, 
                                         (m+1)*self.dy + 2, 0, 25, "black", 
                                         "right"))]
        self._init_length = init_length
        # Create list of locations for each mode
        self.locations = [init_length+50]*self.N
        # Loop over each element in the build spec and add
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
        
        maxloc = max(self.locations)
        for i, loc in enumerate(self.locations):
            if loc < maxloc:
                self._add_wg(loc, (i+1)*self.dy, maxloc-loc)
                self.locations[i] = maxloc
                
        for i in range(self.N):
            self.locations[i] += 50
        
        return
        
    def draw(self) -> draw.Drawing:
        """
        Draws the circuit as defined in the initial class class:
        
        Returns:
        
            draw.Drawing : The created drawing with all circuit components.
        
        """
        extra_lower = 0
        # Create a new drawing of the correct size
        self.d = draw.Drawing(max(self.locations) + 100, 
                              (self.N+1)*self.dy + extra_lower)
        
        # Add frame around circuit
        border = 100
        dx = max(self.locations) - self._init_length
        dy = (self.N-1)*self.dy + border*2 + extra_lower
        self.d.append(draw.Rectangle(self._init_length, self.dy-border, dx, dy,
                                     fill = 'none', stroke = "black"))
        # Add ticks to left side of frame
        length, width = 8, 1
        for i in range(self.N):
            self.d.append(draw.Rectangle(self._init_length-length, 
                                         (i+1)*self.dy-width/2, length, width,
                                         fill = "black"))
        
        # Loop over each element in the drawing spec and add
        for c, data in self.draw_spec:
            if c == "wg":
                self._draw_wg(*data)
            elif c == "ps":
                self._draw_ps(*data)
            elif c == "bs":
                self._draw_bs(*data)
            elif c == "text":
                self._draw_text(*data)
            elif c == "lc":
                self._draw_loss(*data)
            elif c == "mode_swaps":
                self._draw_mode_swaps(*data)
            elif c == "unitary":
                self._draw_unitary(*data)
            elif c == "group":
                self._draw_grouped_circuit(*data)
            else:
                raise DisplayError("Element in draw spec not recognised.")
                
        # Adjust size of figure to meet target scale
        target_scale = min(50+self.N*65, 900)
        #target_scale = max(target_scale, 200)
        self.d.set_pixel_scale(1)
        w,h = self.d.calc_render_size()
        self.d.set_pixel_scale(target_scale/h)

        return self.d
    
    