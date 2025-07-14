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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import drawsvg as draw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from lightworks.sdk.utils.exceptions import DisplayError

from .display_utils import MPLSettings, SVGSettings

# ruff: noqa: D102


@dataclass(slots=True, kw_only=True)
class DrawSpec(ABC):
    """Base class for all draw specs."""

    @abstractmethod
    def draw_svg(self) -> draw.DrawingBasicElement | draw.Group: ...

    @abstractmethod
    def draw_mpl(self, axes: plt.Axes) -> None: ...


@dataclass(slots=True, kw_only=True)
class WaveguideDrawing(DrawSpec):
    """
    Contains components for drawing a waveguide using the different
    visualization methods.
    """

    x: float
    y: float
    length: float
    width: float

    def draw_svg(self) -> draw.DrawingBasicElement:
        return draw.Rectangle(
            self.x,
            self.y - self.width / 2,
            self.length,
            self.width,
            fill="black",
        )

    def draw_mpl(self, axes: plt.Axes) -> None:
        axes.add_patch(
            patches.Rectangle(
                (self.x, self.y - self.width / 2),
                self.length,
                self.width,
                facecolor="black",
            )
        )


@dataclass(slots=True, kw_only=True)
class PhaseShifterDrawing(DrawSpec):
    """
    Contains components for drawing a phase shifter using the different
    visualization methods.
    """

    x: float
    y: float
    size: float
    phase: str | float

    def draw_svg(self) -> draw.Group:
        g = draw.Group()
        g.append(
            CurvedRectangle(
                x=self.x,
                y=self.y - self.size / 2,
                size_x=self.size,
                size_y=self.size,
                radius=5,
                colour="#e8532b",
                outline="black",
            ).draw_svg()
        )
        g.append(
            TextDrawing(
                text="PS",
                x=self.x + self.size / 2,
                y=self.y + 2,
                rotation=0,
                size=SVGSettings.TEXT_SIZE.value,
                colour="white",
                alignment="centred",
            ).draw_svg()
        )
        phi_text = simplify_phase(self.phase)
        g.append(
            TextDrawing(
                text=f"φ = {phi_text}",
                x=self.x + self.size / 2,
                y=self.y + self.size,
                rotation=0,
                size=SVGSettings.S_TEXT_SIZE.value,
                colour="black",
                alignment="centred",
            ).draw_svg()
        )
        return g

    def draw_mpl(self, axes: plt.Axes) -> None:
        CurvedRectangle(
            x=self.x,
            y=self.y - self.size / 2,
            size_x=self.size,
            size_y=self.size,
            radius=0.07,
            colour="#e8532b",
            outline="black",
        ).draw_mpl(axes)
        TextDrawing(
            text="PS",
            x=self.x + self.size / 2,
            y=self.y,
            rotation=0,
            size=MPLSettings.TEXT_SIZE.value,
            colour="white",
            alignment="centred",
        ).draw_mpl(axes)
        phi_text = simplify_phase(self.phase)
        TextDrawing(
            text=f"$\\phi = {phi_text}$",
            x=self.x + self.size / 2,
            y=self.y + self.size / 2 + 0.15,
            rotation=0,
            size=MPLSettings.S_TEXT_SIZE.value,
            colour="black",
            alignment="centred",
        ).draw_mpl(axes)


@dataclass(slots=True, kw_only=True)
class BeamSplitterDrawing(DrawSpec):
    """
    Contains components for drawing a beam splitter using the different
    visualization methods.
    """

    x: float
    y: float
    size_x: float
    size_y: float
    offset_y: float
    reflectivity: str | float
    text_offset: float

    def __post_init__(self) -> None:
        if not isinstance(self.reflectivity, str):
            self.reflectivity = round(self.reflectivity, 4)

    def draw_svg(self) -> draw.Group:
        g = draw.Group()
        g.append(
            CurvedRectangle(
                x=self.x,
                y=self.y - self.offset_y,
                size_x=self.size_x,
                size_y=self.size_y,
                radius=5,
                colour="#3e368d",
                outline="black",
            ).draw_svg()
        )
        g.append(
            TextDrawing(
                text="BS",
                x=self.x + self.size_x / 2,
                y=(self.y + self.size_y / 2 - self.offset_y + self.text_offset),
                rotation=0,
                size=SVGSettings.TEXT_SIZE.value,
                colour="white",
                alignment="centred",
            ).draw_svg()
        )
        g.append(
            TextDrawing(
                text=f"r = {self.reflectivity}",
                x=self.x + self.size_x / 2,
                y=self.y + self.size_y,
                rotation=0,
                size=SVGSettings.S_TEXT_SIZE.value,
                colour="black",
                alignment="centred",
            ).draw_svg()
        )
        return g

    def draw_mpl(self, axes: plt.Axes) -> None:
        CurvedRectangle(
            x=self.x,
            y=self.y - self.offset_y / 2,
            size_x=self.size_x,
            size_y=self.size_y,
            radius=0.07,
            colour="#3e368d",
            outline="black",
        ).draw_mpl(axes)
        TextDrawing(
            text="BS",
            x=self.x + self.size_x / 2,
            y=self.y + 0.5,
            rotation=0,
            size=MPLSettings.TEXT_SIZE.value,
            alignment="centred",
            colour="white",
        ).draw_mpl(axes)
        TextDrawing(
            text=f"$r = ${self.reflectivity}",
            x=self.x + self.size_x / 2,
            y=self.y + self.size_y - self.offset_y / 2 + 0.15,
            rotation=0,
            size=MPLSettings.S_TEXT_SIZE.value,
            alignment="centred",
            colour="black",
        ).draw_mpl(axes)


@dataclass(slots=True, kw_only=True)
class UnitaryDrawing(DrawSpec):
    """
    Contains components for drawing a unitary using the different
    visualization methods.
    """

    x: float
    y: float
    size_x: float
    size_y: float
    offset_y: float
    label: str
    text_size: float

    def draw_svg(self) -> draw.Group:
        g = draw.Group()
        g.append(
            CurvedRectangle(
                x=self.x,
                y=self.y - self.offset_y,
                size_x=self.size_x,
                size_y=self.size_y,
                radius=5,
                colour="#1a0f36",
                outline="black",
            ).draw_svg()
        )
        s = self.text_size * 7 / 5 if len(self.label) == 1 else self.text_size
        r = 270 if len(self.label) > 2 else 0
        g.append(
            TextDrawing(
                text=self.label,
                x=self.x + self.size_x / 2,
                y=self.y + self.size_y / 2 - self.offset_y,
                rotation=r,
                size=s,
                colour="white",
                alignment="centred",
            ).draw_svg()
        )
        return g

    def draw_mpl(self, axes: plt.Axes) -> None:
        CurvedRectangle(
            x=self.x,
            y=self.y - self.offset_y / 2,
            size_x=self.size_x,
            size_y=self.size_y,
            radius=0.07,
            colour="#1a0f36",
            outline="black",
        ).draw_mpl(axes)
        s = 5 / 4 * self.text_size if len(self.label) == 1 else self.text_size
        r = 90 if len(self.label) > 2 else 0
        TextDrawing(
            text=self.label,
            x=self.x + self.size_x / 2,
            y=self.y + (self.size_y - self.offset_y) / 2,
            rotation=r,
            size=s,
            colour="white",
            alignment="centred",
        ).draw_mpl(axes)


@dataclass(slots=True, kw_only=True)
class LossDrawing(DrawSpec):
    """
    Contains components for drawing a loss element using the different
    visualization methods.
    """

    x: float
    y: float
    size: float
    loss: str | float

    def __post_init__(self) -> None:
        if not isinstance(self.loss, str):
            self.loss = str(round(self.loss, 4))

    def draw_svg(self) -> draw.Group:
        g = draw.Group()
        g.append(
            CurvedRectangle(
                x=self.x,
                y=self.y - self.size / 2,
                size_x=self.size,
                size_y=self.size,
                radius=5,
                colour="grey",
                outline="black",
            ).draw_svg()
        )
        g.append(
            TextDrawing(
                text="L",
                x=self.x + self.size / 2,
                y=self.y + 2,
                rotation=0,
                size=SVGSettings.TEXT_SIZE.value,
                colour="white",
                alignment="centred",
            ).draw_svg()
        )
        g.append(
            TextDrawing(
                text=f"loss = {self.loss}",
                x=self.x + self.size / 2,
                y=self.y + self.size,
                rotation=0,
                size=SVGSettings.S_TEXT_SIZE.value,
                colour="black",
                alignment="centred",
            ).draw_svg()
        )
        return g

    def draw_mpl(self, axes: plt.Axes) -> None:
        CurvedRectangle(
            x=self.x,
            y=self.y - self.size / 2,
            size_x=self.size,
            size_y=self.size,
            radius=0.07,
            colour="grey",
            outline="black",
        ).draw_mpl(axes)
        TextDrawing(
            text="L",
            x=self.x + self.size / 2,
            y=self.y,
            rotation=0,
            size=MPLSettings.TEXT_SIZE.value,
            colour="white",
            alignment="centred",
        ).draw_mpl(axes)
        TextDrawing(
            text=f"loss = ${self.loss}$",
            x=self.x + self.size / 2,
            y=self.y + self.size / 2 + 0.15,
            rotation=0,
            size=MPLSettings.S_TEXT_SIZE.value,
            colour="black",
            alignment="centred",
        ).draw_mpl(axes)


@dataclass(slots=True, kw_only=True)
class TextDrawing(DrawSpec):
    """
    Contains components for drawing text using the different visualization
    methods.
    """

    text: str
    x: float
    y: float
    rotation: float
    size: float
    colour: str
    alignment: Literal["left", "centred", "right"]

    def draw_svg(self) -> draw.DrawingBasicElement:
        if self.alignment == "centred":
            ta = "middle"
            db = "middle"
        elif self.alignment == "left":
            ta = "start"
            db = "middle"
        elif self.alignment == "right":
            ta = "end"
            db = "middle"
        else:
            raise DisplayError("Alignment value not recognised.")
        return draw.Text(
            self.text,
            self.size,
            self.x,
            self.y,
            fill=self.colour,
            text_anchor=ta,
            dominant_baseline=db,
            transform=f"rotate({self.rotation}, {self.x}, {self.y})",
        )

    def draw_mpl(self, axes: plt.Axes) -> None:
        if self.alignment == "centred":
            ha = "center"
            va = "center"
        elif self.alignment == "left":
            ha = "left"
            va = "center"
        elif self.alignment == "right":
            ha = "right"
            va = "center"
        else:
            raise DisplayError("Alignment value not recognised.")
        axes.text(
            self.x,
            self.y,
            self.text,
            horizontalalignment=ha,
            verticalalignment=va,
            color=self.colour,
            size=self.size,
            rotation=self.rotation,
        )


@dataclass(slots=True, kw_only=True)
class ModeSwapDrawing(DrawSpec):
    """
    Contains components for drawing mode swaps using the different
    visualization methods.
    """

    x: float
    ys: list[tuple[float, float]]
    size_x: float
    wg_width: float

    def draw_svg(self) -> draw.Group:
        g = draw.Group()
        for y0, y1 in self.ys:
            w = self.wg_width / 2
            m = np.arctan(abs(y1 - y0) / self.size_x)
            if y0 < y1:
                dx1 = w * m
                dx2 = 0
            else:
                dx1 = 0
                dx2 = w * m

            points = [
                self.x + dx1,
                y0 - w,
                self.x,
                y0 - w,
                self.x,
                y0 + w,
                self.x + dx2,
                y0 + w,
                self.x + self.size_x - dx1,
                y1 + w,
                self.x + self.size_x,
                y1 + w,
                self.x + self.size_x,
                y1 - w,
                self.x + self.size_x - dx2,
                y1 - w,
            ]
            poly = draw.Lines(*points, fill="black", close=True)
            g.append(poly)
        return g

    def draw_mpl(self, axes: plt.Axes) -> None:
        for y0, y1 in self.ys:
            w = self.wg_width / 2
            if y0 < y1:
                dx1 = w * np.arctan(abs(y1 - y0) / self.size_x)
                dx2 = 0
            else:
                dx1 = 0
                dx2 = w * np.arctan(abs(y1 - y0) / self.size_x)
            points = [
                (self.x + dx1, y0 - w),
                (self.x, y0 - w),
                (self.x, y0 + w),
                (self.x + dx2, y0 + w),
                (self.x + self.size_x - dx1, y1 + w),
                (self.x + self.size_x, y1 + w),
                (self.x + self.size_x, y1 - w),
                (self.x + self.size_x - dx2, y1 - w),
            ]
            axes.add_patch(patches.Polygon(points, facecolor="black"))


@dataclass(slots=True, kw_only=True)
class HeraldDrawing(DrawSpec):
    """
    Contains components for drawing herald markers using the different
    visualization methods.
    """

    x: float
    y: float
    size: float
    n_photons: int

    def draw_svg(self) -> draw.Group:
        g = draw.Group()
        g.append(
            draw.Circle(
                self.x, self.y, self.size, fill="#3e368d", stroke="black"
            )
        )
        g.append(
            TextDrawing(
                text=str(self.n_photons),
                x=self.x,
                y=self.y + 2.5,
                rotation=0,
                size=30,
                colour="white",
                alignment="centred",
            ).draw_svg()
        )
        return g

    def draw_mpl(self, axes: plt.Axes) -> None:
        axes.add_patch(
            patches.Circle(
                (self.x, self.y),
                self.size,
                facecolor="#3e368d",
                edgecolor="black",
            )
        )
        TextDrawing(
            text=str(self.n_photons),
            x=self.x,
            y=self.y + 0.01,
            rotation=0,
            size=MPLSettings.TEXT_SIZE.value,
            colour="white",
            alignment="centred",
        ).draw_mpl(axes)


@dataclass(slots=True, kw_only=True)
class BarrierDrawing(DrawSpec):
    """
    Contains components for drawing barriers markers using the different
    visualization methods.
    """

    x: float
    y_start: float
    y_end: float
    width: float

    def draw_svg(self) -> draw.DrawingBasicElement:
        return draw.Line(
            self.x,
            self.y_start,
            self.x,
            self.y_end,
            stroke="black",
            stroke_width=self.width,
        )

    def draw_mpl(self, axes: plt.Axes) -> None:
        axes.add_patch(
            patches.Rectangle(
                (self.x - self.width / 2, self.y_start),
                self.width,
                self.y_end - self.y_start,
                facecolor="black",
            )
        )


@dataclass(slots=True, kw_only=True)
class CurvedRectangle(DrawSpec):
    """
    Contains components for drawing curved rectangles using the different
    visualization methods.
    """

    x: float
    y: float
    size_x: float
    size_y: float
    radius: float
    colour: str
    outline: str

    def draw_svg(self) -> draw.DrawingBasicElement:
        return draw.Rectangle(
            self.x,
            self.y,
            self.size_x,
            self.size_y,
            fill=self.colour,
            stroke=self.outline,
            rx=self.radius,
            ry=self.radius,
        )

    def draw_mpl(self, axes: plt.Axes) -> None:
        axes.add_patch(
            patches.FancyBboxPatch(
                (self.x, self.y),
                self.size_x,
                self.size_y,
                boxstyle=f"Round, pad=0, rounding_size={self.radius}",
                facecolor=self.colour,
                edgecolor=self.outline,
            )
        )


def simplify_phase(phase: str | float) -> str:
    """
    Converts a phase values into a fraction of pi if it is some integer amount
    of π/4.
    """
    # Work out value of n*pi/4 closest to phi
    if not isinstance(phase, str):
        n = int(np.round(phase / (np.pi / 4)))
        # Check if value of phi == n*pi/4 to 8 decimal places
        if round(phase, 8) == round(n * np.pi / 4, 8):  # and n > 0:
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
            if phase < 0:
                phi_text = "-" + phi_text
            return phi_text
        # Otherwise round phi to 4 decimal places
        return str(round(phase, 4))
    return phase
