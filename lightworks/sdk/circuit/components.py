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


from dataclasses import dataclass, fields

import numpy as np

from ..utils import check_unitary
from .parameters import Parameter

# ruff: noqa: ANN202, ANN204, D101


class Component:
    def fields(self) -> list:
        """Returns a list of all field from the component dataclass."""
        return [f.name for f in fields(self)]

    def values(self) -> list:
        """Returns a list of all values from the component dataclass."""
        return [getattr(self, f.name) for f in fields(self)]


@dataclass(slots=True)
class BeamSplitter(Component):
    mode_1: int
    mode_2: int
    reflectivity: float | Parameter
    convention: str

    def __post_init__(self):
        # Validate reflectivity
        if not isinstance(self.reflectivity, Parameter):
            if not 0 <= self.reflectivity <= 1:
                raise ValueError("Reflectivity must be in range [0,1].")
        # And check beam splitter convention
        all_convs = ["Rx", "H"]
        if self.convention not in all_convs:
            msg = "Provided beam splitter convention should in ['"
            for conv in all_convs:
                msg += conv + ", "
            msg = msg[:-2] + "']"
            raise ValueError(msg)

    @property
    def _reflectivity(self) -> float:
        if isinstance(self.reflectivity, Parameter):
            return self.reflectivity.get()
        return self.reflectivity


@dataclass(slots=True)
class PhaseShifter(Component):
    mode: int
    phi: float | Parameter

    @property
    def _phi(self) -> float:
        if isinstance(self.phi, Parameter):
            return self.phi.get()
        return self.phi


@dataclass(slots=True)
class Loss(Component):
    mode: int
    loss: float | Parameter

    @property
    def _loss(self) -> float:
        if isinstance(self.loss, Parameter):
            return self.loss.get()
        return self.loss


@dataclass(slots=True)
class Barrier(Component):
    modes: list


@dataclass(slots=True)
class ModeSwaps(Component):
    swaps: dict

    def __post_init__(self):
        # Check swaps are valid
        in_modes = sorted(self.swaps.keys())
        out_modes = sorted(self.swaps.values())
        if in_modes != out_modes:
            raise ValueError(
                "Mode swaps not complete, dictionary keys and values should "
                "contain the same modes."
            )


@dataclass(slots=True)
class Group(Component):
    circuit_spec: list
    name: str
    mode_1: int
    mode_2: int
    heralds: dict


@dataclass(slots=True)
class UnitaryMatrix(Component):
    mode: int
    unitary: np.ndarray
    label: str

    def __post_init__(self):
        # Check type of supplied unitary
        if not isinstance(self.unitary, (np.ndarray, list)):
            raise TypeError("Unitary should be a numpy array or list.")
        self.unitary = np.array(self.unitary)

        # Ensure unitary is valid
        if not check_unitary(self.unitary):
            raise ValueError("Provided matrix is not unitary.")

        # Also check label is a string
        if not isinstance(self.label, str):
            raise TypeError("Label for unitary should be a string.")
