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
from dataclasses import dataclass, field, fields
from numbers import Real
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from lightworks.sdk.utils.matrix import check_unitary
from lightworks.sdk.utils.param_unitary import ParameterizedUnitary
from lightworks.sdk.utils.permutation import permutation_mat_from_swaps_dict

from .parameters import Parameter

SERIALIZED_COMPONENT = tuple[str, dict[str, Any]]

# ruff: noqa: D102


@dataclass(slots=True)
class Component(ABC):
    """
    Generic baseclass for all components. Implements a number of useful methods.
    """

    @abstractmethod
    def get_unitary(self, n_modes: int) -> NDArray[np.complex128]:
        """
        Returns a unitary matrix corresponding to the transformation implemented
        by the component with size n_modes.
        """

    @abstractmethod
    def serialize(self) -> SERIALIZED_COMPONENT:
        """
        Creates a serializable tuple of details for the current component.
        """

    def fields(self) -> list[str]:
        """Returns a list of all field from the component dataclass."""
        return [f.name for f in fields(self)]

    def values(self) -> list[Any]:
        """Returns a list of all values from the component dataclass."""
        return [getattr(self, f.name) for f in fields(self)]


@dataclass(slots=True)
class BeamSplitter(Component):
    """
    Configurable beam splitter element between two assigned modes, with support
    for a number of different conventions.
    """

    mode_1: int
    mode_2: int
    reflectivity: float | Parameter[float]
    convention: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """
        Validates that convention and reflectivity values of the beam splitter
        are valid.
        """
        # Validate reflectivity
        if not is_real(self.reflectivity) and not isinstance(
            self.reflectivity, Parameter
        ):
            raise TypeError(
                "Beam splitter reflectivity should a real number or Parameter."
            )
        if (
            not isinstance(self.reflectivity, Parameter)
            and not 0 <= self.reflectivity <= 1
        ):
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

    def get_unitary(self, n_modes: int) -> NDArray[np.complex128]:
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

    def serialize(self) -> SERIALIZED_COMPONENT:
        return (
            "BeamSplitter",
            {
                "mode_1": self.mode_1,
                "mode_2": self.mode_2,
                "reflectivity": self._reflectivity,
                "convention": self.convention,
            },
        )


@dataclass(slots=True)
class PhaseShifter(Component):
    """
    Implements a phase shift on the assigned mode.
    """

    mode: int
    phi: float | Parameter[float]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Performs validation of all properties of the phase shifters."""
        if not is_real(self.phi) and not isinstance(self.phi, Parameter):
            raise TypeError(
                "Phase shifter value should a real number or Parameter."
            )

    @property
    def _phi(self) -> float:
        if isinstance(self.phi, Parameter):
            return self.phi.get()
        return self.phi

    def get_unitary(self, n_modes: int) -> NDArray[np.complex128]:
        unitary = np.identity(n_modes, dtype=complex)
        unitary[self.mode, self.mode] = np.exp(1j * self._phi)
        return unitary

    def serialize(self) -> SERIALIZED_COMPONENT:
        return ("PhaseShifter", {"mode": self.mode, "phi": self._phi})


@dataclass(slots=True)
class Loss(Component):
    """
    Induces a loss on the selected circuit mode. This requires creation of
    additional loss modes in the unitary matrix.
    """

    mode: int
    loss: float | Parameter[float]

    def validate(self) -> None:
        """Validates loss value is within allowed range."""
        if not 0 <= self._loss <= 1:
            raise ValueError(
                "Provided loss values should be in the range [0,1]."
            )

    @property
    def _loss(self) -> float:
        if isinstance(self.loss, Parameter):
            return self.loss.get()
        return self.loss

    def get_unitary(self, n_modes: int) -> NDArray[np.complex128]:
        self.validate()
        transmission = 1 - self._loss
        # Assumes loss mode to use is last mode in circuit
        unitary = np.identity(n_modes, dtype=complex)
        unitary[self.mode, self.mode] = transmission**0.5
        unitary[n_modes - 1, n_modes - 1] = transmission**0.5
        unitary[self.mode, n_modes - 1] = (1 - transmission) ** 0.5
        unitary[n_modes - 1, self.mode] = (1 - transmission) ** 0.5
        return unitary

    def serialize(self) -> SERIALIZED_COMPONENT:
        return ("Loss", {"mode": self.mode, "loss": self._loss})


@dataclass(slots=True)
class Barrier(Component):
    """
    Adds a barrier across selected circuit modes.
    """

    modes: list[int]

    def get_unitary(self, n_modes: int) -> NDArray[np.complex128]:
        return np.identity(n_modes, dtype=complex)

    def serialize(self) -> SERIALIZED_COMPONENT:
        return ("Barrier", {"modes": list(self.modes)})


@dataclass(slots=True)
class ModeSwaps(Component):
    """
    Performs ideal swaps between selected modes of the circuit.
    """

    swaps: dict[int, int]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Performs validation of all properties of the mode swaps."""
        # Check swaps are valid
        in_modes = sorted(self.swaps.keys())
        out_modes = sorted(self.swaps.values())
        if in_modes != out_modes:
            raise ValueError(
                "Mode swaps not complete, dictionary keys and values should "
                "contain the same modes."
            )

    def get_unitary(self, n_modes: int) -> NDArray[np.complex128]:
        return permutation_mat_from_swaps_dict(self.swaps, n_modes)

    def serialize(self) -> SERIALIZED_COMPONENT:
        return ("ModeSwaps", {"swaps": self.swaps})


@dataclass(slots=True)
class Group(Component):
    """
    Stores a group of components which have been added to a circuit.
    """

    circuit_spec: list[Component]
    name: str
    mode_1: int
    mode_2: int
    heralds: "HeraldData"

    def get_unitary(self, n_modes: int) -> None:  # type: ignore[override] # noqa: ARG002
        return None

    def serialize(self) -> NoReturn:
        raise RuntimeError(
            "Groups must be unpacked before attempting to serialize"
        )


@dataclass(slots=True)
class UnitaryMatrix(Component):
    """
    Implements a unitary transformation across a subset of circuit modes.
    """

    mode: int
    unitary: NDArray[np.complex128] | ParameterizedUnitary
    label: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Performs validation of all properties of the unitary matrix."""
        # Check type of supplied unitary
        if isinstance(self.unitary, np.ndarray | list):
            self.unitary = np.array(self.unitary)

            # Ensure unitary is valid
            if not check_unitary(self.unitary):
                raise ValueError("Provided matrix is not unitary.")
        elif not isinstance(self.unitary, ParameterizedUnitary):
            raise TypeError("Unitary should be a numpy array or list.")

        # Also check label is a string
        if not isinstance(self.label, str):
            raise TypeError("Label for unitary should be a string.")

    def get_unitary(self, n_modes: int) -> NDArray[np.complex128]:
        if isinstance(self.unitary, ParameterizedUnitary):
            sub_unitary = self.unitary.unitary
            if not check_unitary(sub_unitary):
                raise ValueError("Provided matrix is not unitary.")
        else:
            sub_unitary = self.unitary
        unitary = np.identity(n_modes, dtype=complex)
        nm = self.unitary.shape[0]
        unitary[self.mode : self.mode + nm, self.mode : self.mode + nm] = (
            sub_unitary[:, :]
        )
        return unitary

    def serialize(self) -> SERIALIZED_COMPONENT:
        if isinstance(self.unitary, ParameterizedUnitary):
            unitary = self.unitary.unitary
            if not check_unitary(unitary):
                raise ValueError("Provided matrix is not unitary.")
        else:
            unitary = self.unitary
        return (
            "UnitaryMatrix",
            {
                "mode": self.mode,
                "unitary": unitary.astype(str).tolist(),
                "label": self.label,
            },
        )


@dataclass(slots=True, frozen=True)
class HeraldData:
    """Stores the heralding data on a circuit"""

    input: dict[int, int] = field(default_factory=dict)
    output: dict[int, int] = field(default_factory=dict)


def is_real(value: Any) -> bool:
    """General function for checking if a value is a real number."""
    return isinstance(value, Real) and not isinstance(value, bool)
