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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields

import numpy as np

from ..utils import check_unitary, permutation_mat_from_swaps_dict
from .parameters import Parameter

# ruff: noqa: ANN202, ANN204, D101, D102


@dataclass(slots=False)
class Component(metaclass=ABCMeta):
    @abstractmethod
    def get_unitary(self, n_modes: int) -> np.ndarray:
        """Returns unitary matrix for component of size n_modes."""

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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
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

    def get_unitary(self, n_modes: int) -> np.ndarray:
        self.validate()
        theta = np.arccos(self._reflectivity**0.5)
        unitary = np.identity(n_modes, dtype=complex)
        if self.convention == "Rx":
            unitary[self.mode_1, self.mode_1] = np.cos(theta)
            unitary[self.mode_1, self.mode_2] = 1j * np.sin(theta)
            unitary[self.mode_2, self.mode_1] = 1j * np.sin(theta)
            unitary[self.mode_2, self.mode_2] = np.cos(theta)
        elif self.convention == "H":
            unitary[self.mode_1, self.mode_1] = np.cos(theta)
            unitary[self.mode_1, self.mode_2] = np.sin(theta)
            unitary[self.mode_2, self.mode_1] = np.sin(theta)
            unitary[self.mode_2, self.mode_2] = -np.cos(theta)
        return unitary


@dataclass(slots=True)
class PhaseShifter(Component):
    mode: int
    phi: float | Parameter

    @property
    def _phi(self) -> float:
        if isinstance(self.phi, Parameter):
            return self.phi.get()
        return self.phi

    def get_unitary(self, n_modes: int) -> np.ndarray:
        unitary = np.identity(n_modes, dtype=complex)
        unitary[self.mode, self.mode] = np.exp(1j * self._phi)
        return unitary


@dataclass(slots=True)
class Loss(Component):
    mode: int
    loss: float | Parameter

    def validate(self) -> None:
        if self._loss < 0:
            raise ValueError("Provided loss values should be greater than 0.")

    @property
    def _loss(self) -> float:
        if isinstance(self.loss, Parameter):
            return self.loss.get()
        return self.loss

    def get_unitary(self, n_modes: int) -> np.ndarray:
        self.validate()
        transmission = 10 ** (-self._loss / 10)
        # Assumes loss mode to use is last mode in circuit
        unitary = np.identity(n_modes, dtype=complex)
        unitary[self.mode, self.mode] = transmission**0.5
        unitary[n_modes - 1, n_modes - 1] = transmission**0.5
        unitary[self.mode, n_modes - 1] = (1 - transmission) ** 0.5
        unitary[n_modes - 1, self.mode] = (1 - transmission) ** 0.5
        return unitary


@dataclass(slots=True)
class Barrier(Component):
    modes: list

    def get_unitary(self, n_modes: int) -> np.ndarray:
        return np.identity(n_modes, dtype=complex)


@dataclass(slots=True)
class ModeSwaps(Component):
    swaps: dict

    def __post_init__(self) -> None:
        # Check swaps are valid
        in_modes = sorted(self.swaps.keys())
        out_modes = sorted(self.swaps.values())
        if in_modes != out_modes:
            raise ValueError(
                "Mode swaps not complete, dictionary keys and values should "
                "contain the same modes."
            )

    def get_unitary(self, n_modes: int) -> np.ndarray:
        return permutation_mat_from_swaps_dict(self.swaps, n_modes)


@dataclass(slots=True)
class Group(Component):
    circuit_spec: list
    name: str
    mode_1: int
    mode_2: int
    heralds: dict

    def get_unitary(self, n_modes: int) -> None:  # type: ignore[override] # noqa: ARG002
        return None


@dataclass(slots=True)
class UnitaryMatrix(Component):
    mode: int
    unitary: np.ndarray
    label: str

    def __post_init__(self) -> None:
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

    def get_unitary(self, n_modes: int) -> np.ndarray:
        unitary = np.identity(n_modes, dtype=complex)
        nm = self.unitary.shape[0]
        unitary[self.mode : self.mode + nm, self.mode : self.mode + nm] = (
            self.unitary[:, :]
        )
        return unitary
