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

import matplotlib.patches as patches
import numpy as np

class DisplayComponentsMPL:
    
    def _add_wg(self, x: float, y: float, length: float) -> None:
        """Add a waveguide to the axis."""
        # Create rectangle using provided length and waveguide width and then
        # add patch
        rect = patches.Rectangle((x, y-self.wg_width/2), length, self.wg_width, 
                                 facecolor = "black")
        self.ax.add_patch(rect)
        
        return
    
    def _add_ps(self, mode: int, phi: float) -> None:
        """Add a phase shifter to the axis."""
        # Set size of phase shifter box and length of connector
        size = 0.5
        con_length = 0.5
        # Get x and y locs of target modes
        xloc = self.locations[mode]
        yloc = mode
        # Add input waveguides
        self._add_wg(xloc, mode, con_length)
        xloc += con_length
        # Add phase shifter square
        rect = patches.Rectangle((xloc, yloc-size/2), size, size, 
                                 facecolor = "#e8532b", edgecolor = "black")
        self.ax.add_patch(rect)
        # Include text to label applied phase shift
        self.ax.text(xloc + size/2, yloc, "PS", horizontalalignment = "center", 
                     verticalalignment = "center", color= "white", size = 8)
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
            phi_text = f"$\\phi = {phi_text}$"
        else:
            phi_text = "$\\phi =$ " + phi
        self.ax.text(xloc + size/2, yloc + size/2 + 0.15, 
                     phi_text, color= "black", size = 5,
                     horizontalalignment = "center", 
                     verticalalignment = "center")
        xloc += size
        # Add output waveguides
        self._add_wg(xloc, mode, con_length)
        # Update mode locations list
        self.locations[mode] = xloc + con_length
        
        return
    
    def _add_bs(self, mode1: int, mode2: int, ref: float) -> None:
        """Add a beam splitter across to provided modes to the axis."""
        size_x = 0.5 # x beam splitter size
        con_length = 0.5 # input/output waveguide length
        offset = 0.5 # Offset of beam splitter shape from mode centres
        size_y = offset + abs(mode2 - mode1) # Find y size
        # Get x and y locations
        yloc = min(mode1, mode2)
        xloc = max(self.locations[mode1:mode2+1])
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.locations[mode1:mode2+1]):
            if loc < xloc:
                self._add_wg(loc, yloc + i, xloc - loc)
        # Add input waveguides for all included modes
        modes = range(min(mode1, mode2), max(mode1, mode2) + 1, 1)
        for i in modes:
            self._add_wg(xloc, i, con_length)
        xloc += con_length
        # Add beam splitter rectangle shape
        rect = patches.Rectangle((xloc, yloc-offset/2), size_x, size_y, 
                                 facecolor = "#3e368d", alpha = 1, 
                                 edgecolor = "black")
        self.ax.add_patch(rect)
        # Label beam splitter as BS and display reflectivity
        self.ax.text(xloc + size_x/2, yloc+0.5, "BS", size= 8,
                     horizontalalignment = "center", 
                     verticalalignment = "center", color= "white")
        if not isinstance(ref, str):
            ref = round(ref, 4)
        self.ax.text(xloc + size_x/2, yloc + size_y - offset/2 + 0.15, 
                     f"$r =$ {ref}", color= "black", size = 5,
                     horizontalalignment = "center", 
                     verticalalignment = "center")
        # For any modes in between the beam splitter modes add a waveguide 
        # across the beam splitter
        if mode2 - mode1 > 1:
            for i in range(mode1+1, mode2):
                self._add_wg(xloc, i, size_x)
        xloc += size_x
        # Add output waveguides
        for i in modes:
            self._add_wg(xloc, i, con_length)
        # Update mode locations
        for i in modes:
            self.locations[i] = xloc + con_length
            
        return
    
    def _add_loss(self, mode: int, loss: float) -> None:
        """Add a loss channel to the specified mode"""
        # Set size of loss element and input/output waveguide length
        size = 0.5
        con_length = 0.5
        # Get x and y locations
        xloc = self.locations[mode]
        yloc = mode
        # Add an input waveguide
        self._add_wg(xloc, mode, con_length)
        xloc += con_length
        # Add loss elements
        rect = patches.Rectangle((xloc, yloc-size/2), size, size, 
                                 facecolor = "grey", edgecolor = "black")
        self.ax.add_patch(rect)
        # Label element and add text will loss amount in dB
        self.ax.text(xloc + size/2, yloc, "L", horizontalalignment = "center", 
                     verticalalignment = "center", color= "white", size = 8)
        if not isinstance(loss, str):
            loss_text = f"$loss = {round(loss,4)} dB$"
        else:
            loss_text = "$loss =$ " + loss
        self.ax.text(xloc + size/2, yloc + size/2 + 0.15, 
                     loss_text, color = "black", 
                     size = 5, horizontalalignment = "center", 
                     verticalalignment = "center")
        xloc += size
        # Add output waveguide
        self._add_wg(xloc, mode, con_length)
        # Update mode position
        self.locations[mode] = xloc + con_length
        
        return
    
    def _add_barrier(self, modes: list) -> None:
        """
        Add a barrier which will separate different parts of the circuit. This
        is applied to the provided modes.
        """
        max_loc = 0
        for m in modes:
            max_loc = max(max_loc, self.locations[m])
        for m in modes:
            loc = self.locations[m]
            if loc < max_loc:
                self._add_wg(loc, m, max_loc - loc)
            self.locations[m] = max_loc
        
        return
    
    def _add_mode_swaps(self, swaps: dict) -> None:
        """Add mode swaps between provided modes to the axis."""
        size_x = 1 # x beam splitter size
        con_length = 0.25 # input/output waveguide length
        min_mode = min(swaps)
        max_mode = max(swaps)
        # Get x and y locations
        xloc = max(self.locations[min_mode:max_mode+1])
        ylocs = []
        for i, j in swaps.items():
            ylocs.append((i, j))
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.locations[min_mode:max_mode+1]):
            if loc < xloc:
                self._add_wg(loc, min_mode + i, xloc - loc)
        # Add input waveguides for all included modes
        modes = range(min_mode, max_mode + 1, 1)
        for i in modes:
            self._add_wg(xloc, i, con_length)
        xloc += con_length
        for y0, y1 in ylocs:
            w = self.wg_width/2
            if y0 < y1:
                dx1 = w*np.arctan(abs(y1-y0)/size_x)
                dx2 = 0
            else:
                dx1 = 0
                dx2 = w*np.arctan(abs(y1-y0)/size_x)
            points = [(xloc+dx1, y0-w), (xloc, y0-w), (xloc, y0+w), 
                      (xloc+dx2, y0+w), (xloc+size_x-dx1, y1+w), 
                      (xloc+size_x, y1+w), (xloc+size_x, y1-w),
                      (xloc+size_x-dx2, y1-w)]
            poly = patches.Polygon(points, facecolor = "black")
            self.ax.add_patch(poly)
        xloc += size_x
        # Add output waveguides
        for i in modes:
            self._add_wg(xloc, i, con_length)
        # Update mode locations
        for i in modes:
            self.locations[i] = xloc + con_length
            
        return
    
    def _add_unitary(self, mode1: int, mode2: int) -> None:
        """Add a unitary representation to the axis."""
        size_x = 1 # Unitary x size
        con_length = 0.5 # Input/output waveguide lengths
        offset = 0.5 # Offset of unitary square from modes
        size_y = offset + abs(mode2 - mode1) # Find total unitary size
        # Get x and y positions
        yloc = min(mode1, mode2)
        xloc = max(self.locations[mode1:mode2+1])
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.locations[mode1:mode2+1]):
            if loc < xloc:
                self._add_wg(loc, yloc + i, xloc - loc)
        # Add input waveguides
        modes = range(min(mode1, mode2), max(mode1, mode2)+1, 1)
        for i in modes:
            self._add_wg(xloc, i, con_length)
        xloc += con_length
        # Add unitary shape and label
        rect = patches.Rectangle((xloc, yloc-offset/2), size_x, size_y, 
                                 facecolor = "#1a0f36", 
                                 edgecolor = "black")
        self.ax.add_patch(rect)
        self.ax.text(xloc + size_x/2, yloc + abs(mode2 - mode1)/2, 
                     "U", horizontalalignment = "center", size= 8,
                     verticalalignment = "center", color= "white")
        xloc += size_x
        # Add output waveguides
        for i in modes:
            self._add_wg(xloc, i, con_length)
        # Update mode positions
        for i in modes:
            self.locations[i] = xloc + con_length
            
        return
    
    def _add_grouped_circuit(self, mode1: int, mode2: int, name: str) -> None:
        """Add a grouped circuit drawing to the axis."""
        size_x = 1 # x size
        con_length = 0.5 # Input/output waveguide lengths
        offset = 0.5 # Offset of square from modes
        size_y = offset + abs(mode2 - mode1) # Find total unitary size
        # Get x and y positions
        yloc = min(mode1, mode2)
        xloc = max(self.locations[mode1:mode2+1])
        # Add initial connectors for any modes which haven't reach xloc yet:
        for i, loc in enumerate(self.locations[mode1:mode2+1]):
            if loc < xloc:
                self._add_wg(loc, yloc + i, xloc - loc)
        # Add input waveguides
        modes = range(min(mode1, mode2), max(mode1, mode2)+1, 1)
        for i in modes:
            self._add_wg(xloc, i, con_length)
        xloc += con_length
        # Add circuit shape and label
        rect = patches.Rectangle((xloc, yloc-offset/2), size_x, size_y, 
                                 facecolor = "#1a0f36", 
                                 edgecolor = "black")
        self.ax.add_patch(rect)
        s = 10 if len(name) == 1 else 8
        r = 90 if len(name) > 2 else 0
        self.ax.text(xloc + size_x/2, yloc + abs(mode2 - mode1)/2, 
                     name, horizontalalignment = "center", size = s,
                     verticalalignment = "center", color = "white", 
                     rotation = r)
        xloc += size_x
        # Add output waveguides
        for i in modes:
            self._add_wg(xloc, i, con_length)
        # Update mode positions
        for i in modes:
            self.locations[i] = xloc + con_length
            
        return