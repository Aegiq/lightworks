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

from typing import Any

import drawsvg as draw
import numpy as np

from ..utils import DisplayError


class DrawSVGComponents:
    """
    Manages addition of all components to the Drawing.
    """

    def __init__(self, drawing: draw.Drawing, wg_width: float) -> None:
        self.d = drawing
        self.wg_width = wg_width

        return

    def add(self, draw_spec: list[tuple[str, tuple[Any, ...]]]) -> None:
        """
        Adds components to the provided drawing using the draw spec.
        """
        # Loop over each element in the drawing spec and add
        for c, data in draw_spec:
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
            elif c == "herald":
                self._draw_herald(*data)
            else:
                raise DisplayError("Element in draw spec not recognised.")

    def _draw_wg(self, x: float, y: float, length: float) -> None:
        r = draw.Rectangle(
            x, y - self.wg_width / 2, length, self.wg_width, fill="black"
        )
        self.d.append(r)
        return

    def _draw_ps(self, x: float, y: float, size: float) -> None:
        r = draw.Rectangle(
            x,
            y - size / 2,
            size,
            size,
            fill="#e8532b",
            stroke="black",
            rx=5,
            ry=5,
        )
        self.d.append(r)
        return

    def _draw_bs(
        self, x: float, y: float, size_x: float, size_y: float, offset_y: float
    ) -> None:
        r = draw.Rectangle(
            x,
            y - offset_y,
            size_x,
            size_y,
            fill="#3e368d",
            stroke="black",
            rx=5,
            ry=5,
        )
        self.d.append(r)
        return

    def _draw_unitary(
        self, x: float, y: float, size_x: float, size_y: float, offset_y: float
    ) -> None:
        r = draw.Rectangle(
            x,
            y - offset_y,
            size_x,
            size_y,
            fill="#1a0f36",
            stroke="black",
            rx=5,
            ry=5,
        )
        self.d.append(r)
        return

    def _draw_loss(self, x: float, y: float, size: float) -> None:
        r = draw.Rectangle(
            x, y - size / 2, size, size, fill="grey", stroke="black", rx=5, ry=5
        )
        self.d.append(r)
        return

    def _draw_text(
        self,
        text: str,
        x: float,
        y: float,
        rotation: float,
        size: float,
        colour: str,
        alignment: str,
    ) -> None:
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
        t = draw.Text(
            text,
            size,
            x,
            y,
            fill=colour,
            text_anchor=ta,
            dominant_baseline=db,
            transform=f"rotate({rotation}, {x}, {y})",
        )
        self.d.append(t)
        return

    def _draw_mode_swaps(
        self, x: float, ys: list[tuple[float, float]], size_x: float
    ) -> None:
        for y0, y1 in ys:
            w = self.wg_width / 2
            m = np.arctan(abs(y1 - y0) / size_x)
            if y0 < y1:
                dx1 = w * m
                dx2 = 0
            else:
                dx1 = 0
                dx2 = w * m

            points = [
                x + dx1,
                y0 - w,
                x,
                y0 - w,
                x,
                y0 + w,
                x + dx2,
                y0 + w,
                x + size_x - dx1,
                y1 + w,
                x + size_x,
                y1 + w,
                x + size_x,
                y1 - w,
                x + size_x - dx2,
                y1 - w,
            ]
            poly = draw.Lines(*points, fill="black", close=True)
            self.d.append(poly)

        return

    def _draw_grouped_circuit(
        self, x: float, y: float, size_x: float, size_y: float, offset_y: float
    ) -> None:
        r = draw.Rectangle(
            x,
            y - offset_y,
            size_x,
            size_y,
            fill="#1a0f36",
            stroke="black",
            rx=5,
            ry=5,
        )
        self.d.append(r)
        return

    def _draw_herald(self, x: float, y: float, size: float) -> None:
        c = draw.Circle(x, y, size, fill="#3e368d", stroke="black")
        self.d.append(c)
        return
