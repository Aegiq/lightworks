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

import drawsvg as draw
import numpy as np

from lightworks.sdk.utils.exceptions import DisplayError

# ruff: noqa: D102


@dataclass(slots=True, kw_only=True)
class DrawSpec(ABC):
    """Base class for all draw specs."""

    @abstractmethod
    def draw_svg(self) -> list[draw.DrawingBasicElement]: ...


@dataclass(slots=True, kw_only=True)
class WaveguideDrawing(DrawSpec):
    """
    Desc
    """

    x: float
    y: float
    length: float
    width: float

    def draw_svg(self) -> list[draw.DrawingBasicElement]:
        return [
            draw.Rectangle(
                self.x,
                self.y - self.width / 2,
                self.length,
                self.width,
                fill="black",
            )
        ]


@dataclass(slots=True, kw_only=True)
class PhaseShifterDrawing(DrawSpec):
    """
    Desc
    """

    x: float
    y: float
    size: float

    def draw_svg(self) -> list[draw.DrawingBasicElement]:
        return [
            draw.Rectangle(
                self.x,
                self.y - self.size / 2,
                self.size,
                self.size,
                fill="#e8532b",
                stroke="black",
                rx=5,
                ry=5,
            )
        ]


@dataclass(slots=True, kw_only=True)
class BeamSplitterDrawing(DrawSpec):
    """
    Desc
    """

    x: float
    y: float
    size_x: float
    size_y: float
    offset_y: float

    def draw_svg(self) -> list[draw.DrawingBasicElement]:
        return [
            draw.Rectangle(
                self.x,
                self.y - self.offset_y,
                self.size_x,
                self.size_y,
                fill="#3e368d",
                stroke="black",
                rx=5,
                ry=5,
            )
        ]


@dataclass(slots=True, kw_only=True)
class UnitaryDrawing(DrawSpec):
    """
    Desc
    """

    x: float
    y: float
    size_x: float
    size_y: float
    offset_y: float

    def draw_svg(self) -> list[draw.DrawingBasicElement]:
        return [
            draw.Rectangle(
                self.x,
                self.y - self.offset_y,
                self.size_x,
                self.size_y,
                fill="#1a0f36",
                stroke="black",
                rx=5,
                ry=5,
            )
        ]


@dataclass(slots=True, kw_only=True)
class LossDrawing(DrawSpec):
    """
    Desc
    """

    x: float
    y: float
    size: float

    def draw_svg(self) -> list[draw.DrawingBasicElement]:
        return [
            draw.Rectangle(
                self.x,
                self.y - self.size / 2,
                self.size,
                self.size,
                fill="grey",
                stroke="black",
                rx=5,
                ry=5,
            )
        ]


@dataclass(slots=True, kw_only=True)
class TextDrawing(DrawSpec):
    """
    Desc
    """

    text: str
    x: float
    y: float
    rotation: float
    size: float
    colour: str
    alignment: str

    def draw_svg(self) -> list[draw.DrawingBasicElement]:
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
        return [
            draw.Text(
                self.text,
                self.size,
                self.x,
                self.y,
                fill=self.colour,
                text_anchor=ta,
                dominant_baseline=db,
                transform=f"rotate({self.rotation}, {self.x}, {self.y})",
            )
        ]


@dataclass(slots=True, kw_only=True)
class ModeSwapDrawing(DrawSpec):
    """
    Desc
    """

    x: float
    ys: list[tuple[float, float]]
    size_x: float
    wg_width: float

    def draw_svg(self) -> list[draw.DrawingBasicElement]:
        drawings = []
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
            drawings.append(poly)
        return drawings


@dataclass(slots=True, kw_only=True)
class GroupedCircuitDrawing(DrawSpec):
    """
    Desc
    """

    x: float
    y: float
    size_x: float
    size_y: float
    offset_y: float

    def draw_svg(self) -> list[draw.DrawingBasicElement]:
        return [
            draw.Rectangle(
                self.x,
                self.y - self.offset_y,
                self.size_x,
                self.size_y,
                fill="#1a0f36",
                stroke="black",
                rx=5,
                ry=5,
            )
        ]


@dataclass(slots=True, kw_only=True)
class HeraldDrawing(DrawSpec):
    """
    Desc
    """

    x: float
    y: float
    size: float

    def draw_svg(self) -> list[draw.DrawingBasicElement]:
        return [
            draw.Circle(
                self.x, self.y, self.size, fill="#3e368d", stroke="black"
            )
        ]
