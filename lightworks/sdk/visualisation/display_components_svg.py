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

from ..utils import DisplayError

import drawsvg as draw
import numpy as np

class SVGDrawSpec:
    
    def _add_wg(self, x: float, y: float, length: float) -> None:
        """Add a waveguide to the drawing."""
        # Add a waveguide of the required length to the drawing spec
        self.draw_spec += [("wg", (x-0.05, y, length+0.1))]
        
        return
    
    def _add_ps(self, mode: int, phi: float) -> None:
        """Add a phase shifter to the drawing."""
        size = 50
        con_length = 50
        # Get current x and y locations for the mode
        xloc = self.locations[mode]
        yloc = (mode+1)*self.dy
        # Input waveguide
        self._add_wg(xloc, yloc, con_length)
        xloc += con_length
        # Add phase shifter section
        self.draw_spec += [("ps", (xloc, yloc, size))]
        self.draw_spec += [("text", ("PS", xloc+size/2, yloc+2, 0, 25, "white",
                                     "centred"))]
        # Work out value of n*pi/4 closest to phi
        if not isinstance(phi, str):
            n = int(np.round(phi/(np.pi/4)))
            # Check if value of phi == n*pi/4 to 8 decimal places
            if round(phi, 8) == round(n*np.pi/4, 8):# and n > 0:
                n = abs(n)
                # Set text with either pi or pi/2 or pi/4
                if n == 0:
                    phi_text = "0"
                elif n%4 == 0:
                    phi_text = str(int(n/4)) + "π" if n > 4 else "π"
                elif n%4 == 2:
                    phi_text = str(int(n/2)) + "π/2" if n > 2 else "π/2"
                else:
                    phi_text = str(int(n)) + "π/4" if n > 1 else "π/4"
                if phi < 0:
                    phi_text = "-" + phi_text
            # Otherwise round phi to 4 decimal places
            else:
                phi_text = round(phi,4)
        else:
            phi_text = phi
        self.draw_spec += [("text", (f"φ = {phi_text}", xloc+size/2, 
                                     yloc+size, 0, 18, "black", "centred"))]
        xloc += size
        # Output waveguide
        self._add_wg(xloc, yloc, con_length)
        self.locations[mode] = xloc + con_length
        
        return
        
    def _add_bs(self, mode1: int, mode2: int, ref: float) -> None:
        """Add a beam splitter across to provided modes to the drawing."""
        size_x = 50 # x beam splitter size
        con_length = 50 # input/output waveguide length
        offset = 50 # Offset of beam splitter shape from mode centres
        size_y = offset + abs(mode2 - mode1)*self.dy
        # Get x and y locations
        yloc = (min(mode1, mode2)+1)*self.dy
        xloc = max(self.locations[mode1:mode2+1])
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.locations[mode1:mode2+1]):
            if loc < xloc:
                self._add_wg(loc, yloc + i*self.dy, xloc - loc)
        # Add input waveguides for all included modes
        modes = range(min(mode1, mode2), max(mode1, mode2) + 1, 1)
        for i in modes:
            self._add_wg(xloc, (i+1)*self.dy, con_length)
        xloc += con_length
        # Add beam splitter section
        dy = - 50 if abs(mode2-mode1)%2 == 0 else 0
        self.draw_spec += [("bs", (xloc, yloc, size_x, size_y, offset/2))]
        self.draw_spec += [("text", ("BS", xloc+size_x/2, 
                                     (yloc + (size_y-offset)/2 + dy), 0,
                                     25, "white", "centred"))]
        if not isinstance(ref, str):
            ref = round(ref,4)
        self.draw_spec += [("text", (f"r = {ref}", xloc+size_x/2, 
                                     yloc + size_y, 0, 18, "black", 
                                     "centred"))]
        # For any modes in between the beam splitter modes add a waveguide 
        # across the beam splitter
        if mode2 - mode1 > 1:
            for i in range(mode1+1, mode2):
                self._add_wg(xloc, (i+1)*self.dy, size_x)
        xloc += size_x
        # Add output waveguides
        for i in modes:
            self._add_wg(xloc, (i+1)*self.dy, con_length)
        # Update mode locations
        for i in modes:
            self.locations[i] = xloc + con_length
        
        return
    
    def _add_unitary(self, mode1: int, mode2: int) -> None:
        """Add a unitary matrix representation to the drawing."""
        size_x = 100 # Unitary x size
        con_length = 50 # Input/output waveguide lengths
        offset = 50 # Offset of unitary square from modes
        size_y = offset + abs(mode2 - mode1)*self.dy # Find total size
        # Get x and y positions
        yloc = (min(mode1, mode2)+1)*self.dy
        xloc = max(self.locations[mode1:mode2+1])
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.locations[mode1:mode2+1]):
            if loc < xloc:
                self._add_wg(loc, yloc + i*self.dy, xloc - loc)
        # Add input waveguides
        modes = range(min(mode1, mode2), max(mode1, mode2)+1, 1)
        for i in modes:
            self._add_wg(xloc, (i+1)*self.dy, con_length)
        xloc += con_length
        # Add unitary shape and U label
        self.draw_spec += [("unitary", (xloc, yloc, size_x, size_y, offset/2))]
        self.draw_spec += [("text", (f"U", xloc+size_x/2, 
                                     yloc + (size_y-offset)/2, 0, 35, 
                                     "white", "centred"))]
        xloc += size_x
        # Add output waveguides and update mode positions
        for i in modes:
            self._add_wg(xloc, (i+1)*self.dy, con_length)
            self.locations[i] = xloc + con_length
            
        return
    
    def _add_loss(self, mode: int, loss: float) -> None:
        """Add a loss channel to the specified mode."""
        size = 50
        con_length = 50
        # Get current x and y locations for the mode
        xloc = self.locations[mode]
        yloc = (mode+1)*self.dy
        # Input waveguide
        self._add_wg(xloc, yloc, con_length)
        xloc += con_length
        # Add phase shifter section
        self.draw_spec += [("lc", (xloc, yloc, size))]
        self.draw_spec += [("text", ("L", xloc+size/2, yloc+2, 0, 25, "white",
                                     "centred"))]
        # Add loss label
        if not isinstance(loss, str):
            loss = str(round(loss, 4)) + " dB"
        self.draw_spec += [("text", (f"loss = {loss}", 
                                     xloc+size/2, yloc+size, 0, 18, "black",
                                     "centred"))]
        xloc += size
        # Output waveguide
        self._add_wg(xloc, yloc, con_length)
        self.locations[mode] = xloc + con_length
        
        return
    
    def _add_barrier(self, modes: list) -> None:
        """
        Add a barrier which will separate different parts of the circuit. This
        is applied to the provided modes.
        """
        max_loc = max([self.locations[m] for m in modes])
        for m in modes:
            loc = self.locations[m]
            if loc < max_loc:
                self._add_wg(loc, (m+1)*self.dy, max_loc - loc)
            self.locations[m] = max_loc
        
        return
    
    def _add_mode_swaps(self, swaps: dict) -> None:
        """Add mode swaps between provided modes to the drawing."""
        con_length = 25 # input/output waveguide length
        min_mode = min(swaps)
        max_mode = max(swaps)
        size_x = 50+20*(max_mode-min_mode) # x length of swap element
        # Add in missing mode for swap
        for m in range(min_mode, max_mode+1):
            if m not in swaps:
                swaps[m] = m
        # Get x and y locations
        xloc = max(self.locations[min_mode:max_mode+1])
        ylocs = []
        for i, j in swaps.items():
            ylocs.append(((i+1)*self.dy, (j+1)*self.dy))
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.locations[min_mode:max_mode+1]):
            if loc < xloc:
                self._add_wg(loc, (min_mode+i+1)*self.dy, xloc - loc)
        # Add input waveguides for all included modes
        modes = range(min_mode, max_mode+1, 1)
        for i in modes:
            self._add_wg(xloc, (i+1)*self.dy, con_length)
        xloc += con_length
        # Add beam splitter section
        self.draw_spec += [("mode_swaps", (xloc, ylocs, size_x))]
        xloc += size_x
        # Add output waveguides
        for i in modes:
            self._add_wg(xloc, (i+1)*self.dy, con_length)
        # Update mode locations
        for i in modes:
            self.locations[i] = xloc + con_length
        
        return
    
    def _add_grouped_circuit(self, mode1: int, mode2: int, name: str) -> None:
        """Add a grouped_circuit representation to the drawing."""
        size_x = 100 # Drawing x size
        con_length = 50 # Input/output waveguide lengths
        offset = 50 # Offset of square from modes
        size_y = offset + abs(mode2 - mode1)*self.dy # Find total size
        # Get x and y positions
        yloc = (min(mode1, mode2)+1)*self.dy
        xloc = max(self.locations[mode1:mode2+1])
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.locations[mode1:mode2+1]):
            if loc < xloc:
                self._add_wg(loc, yloc + i*self.dy, xloc - loc)
        # Add input waveguides
        modes = range(min(mode1, mode2), max(mode1, mode2)+1, 1)
        for i in modes:
            self._add_wg(xloc, (i+1)*self.dy, con_length)
        xloc += con_length
        # Add unitary shape and U label
        self.draw_spec += [("group", (xloc, yloc, size_x, size_y, offset/2))]
        s = 25 if self.N > 2 else 20
        s = 35 if len(name) == 1 else s
        r = 270 if len(name) > 2 else 0
        self.draw_spec += [("text", (name, xloc+size_x/2, 
                                     yloc + (size_y-offset)/2, r, s, 
                                     "white", "centred"))]
        xloc += size_x
        # Add output waveguides and update mode positions
        for i in modes:
            self._add_wg(xloc, (i+1)*self.dy, con_length)
            self.locations[i] = xloc + con_length
            
        return
    
class DisplayComponentsSVG:
    
    def _draw_wg(self, x: float, y: float, length: float) -> None:
        
        r = draw.Rectangle(x, y-self.wg_width/2, length, self.wg_width, 
                           fill = "black")
        self.d.append(r)
        return
        
    def _draw_ps(self, x: float, y: float, size: float) -> None:

        r = draw.Rectangle(x, y-size/2, size, size, fill = "#e8532b", 
                           stroke = "black", rx = 5, ry = 5)
        self.d.append(r)
        return
        
    def _draw_bs(self, x: float, y: float, size_x: float, size_y: float, 
                 offset_y: float) -> None:
        
        r = draw.Rectangle(x, y-offset_y, size_x, size_y, fill = "#3e368d", 
                           stroke = "black", rx = 5, ry = 5)
        self.d.append(r)
        return
    
    def _draw_unitary(self, x: float, y: float, size_x: float, size_y: float, 
                      offset_y: float) -> None:
        
        r = draw.Rectangle(x, y-offset_y, size_x, size_y, fill = "#1a0f36", 
                           stroke = "black", rx = 5, ry = 5)
        self.d.append(r)
        return
        
    def _draw_loss(self, x: float, y: float, size: float) -> None:

        r = draw.Rectangle(x, y-size/2, size, size, fill = "grey", 
                           stroke = "black", rx = 5, ry = 5)
        self.d.append(r)
        return
        
    def _draw_text(self, text: str, x: float, y: float, rotation: float, 
                   size: float, colour: str, alignment: str) -> None:
        
        if alignment == "centred":
            ta = "middle"
            db = "middle"
        elif alignment == "left":
            ta = "start"
            db = "middle"
        elif alignment == "right":
            ta = "end"
            db = "middle"
        else:
            raise DisplayError("Alignment value not recognised.")
        t = draw.Text(text, size, x, y, fill = colour, text_anchor = ta,
                      dominant_baseline = db, 
                      transform = f"rotate({rotation}, {x}, {y})")
        self.d.append(t)
        return
    
    def _draw_mode_swaps(self, x, ys, size_x):
        
        for y0, y1 in ys:
            w = self.wg_width/2
            m = np.arctan(abs(y1-y0)/size_x)
            if y0 < y1:
                dx1 = w*m
                dx2 = 0
            else:
                dx1 = 0
                dx2 = w*m
            
            points = [x+dx1, y0-w, x, y0-w, x, y0+w, x+dx2, y0+w, 
                      x+size_x-dx1, y1+w, x+size_x, y1+w, x+size_x, y1-w,
                      x+size_x-dx2, y1-w]
            poly = draw.Lines(*points, fill = "black", close = True)
            self.d.append(poly)
        
        return
    
    def _draw_grouped_circuit(self, x: float, y: float, size_x: float, 
                              size_y: float, offset_y: float) -> None:
        
        r = draw.Rectangle(x, y-offset_y, size_x, size_y, fill = "#1a0f36", 
                           stroke = "black", rx = 5, ry = 5)
        self.d.append(r)
        return