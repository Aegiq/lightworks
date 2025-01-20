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

"""
Contains a variety of single qubit components which act across a pair of
adjacent dual-rail encoded modes, assuming 0 is the first mode and 1 is the
second mode.
"""

import numpy as np

from ...sdk.circuit import Unitary


class I(Unitary):  # noqa: E742
    """
    Implements the identity gate across a pair of modes corresponding to a
    dual-rail encoded qubit.
    """

    def __init__(self) -> None:
        unitary = np.array([[1, 0], [0, 1]])
        super().__init__(unitary, "I")


class H(Unitary):
    """
    Implements a Hadamard across a pair of modes corresponding to a dual-rail
    encoded qubit.
    """

    def __init__(self) -> None:
        unitary = np.array([[1, 1], [1, -1]], dtype=np.complex128) / 2**0.5
        super().__init__(unitary, "H")


class X(Unitary):
    """
    Implements an X gate across a pair of modes corresponding to a dual-rail
    encoded qubit.
    """

    def __init__(self) -> None:
        unitary = np.array([[0, 1], [1, 0]])
        super().__init__(unitary, "X")


class Y(Unitary):
    """
    Implements a Y gate across a pair of modes corresponding to a dual-rail
    encoded qubit.
    """

    def __init__(self) -> None:
        unitary = np.array([[0, -1j], [1j, 0]])
        super().__init__(unitary, "Y")


class Z(Unitary):
    """
    Implements a Z gate across a pair of modes corresponding to a dual-rail
    encoded qubit.
    """

    def __init__(self) -> None:
        unitary = np.array([[1, 0], [0, -1]])
        super().__init__(unitary, "Z")


class S(Unitary):
    """
    Implements an S gate across a pair of modes corresponding to a dual-rail
    encoded qubit.
    """

    def __init__(self) -> None:
        unitary = np.array([[1, 0], [0, 1j]])
        super().__init__(unitary, "S")


class Sadj(Unitary):
    """
    Implements the adjoint S gate across a pair of modes corresponding to a
    dual-rail encoded qubit.
    """

    def __init__(self) -> None:
        unitary = np.array([[1, 0], [0, -1j]])
        super().__init__(unitary, "S_adj")


class T(Unitary):
    """
    Implements a T gate across a pair of modes corresponding to a dual-rail
    encoded qubit.
    """

    def __init__(self) -> None:
        unitary = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        super().__init__(unitary, "T")


class Tadj(Unitary):
    """
    Implements the adjoint T gate across a pair of modes corresponding to a
    dual-rail encoded qubit.
    """

    def __init__(self) -> None:
        unitary = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])
        super().__init__(unitary, "T_adj")


class SX(Unitary):
    """
    Implements a SX gate across a pair of modes corresponding to a dual-rail
    encoded qubit.
    """

    def __init__(self) -> None:
        unitary = 0.5 * np.array(
            [[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128
        )
        super().__init__(unitary, "SX")


class P(Unitary):
    """
    Implements a phase gate with phase theta across a pair of modes
    corresponding to a dual-rail encoded qubit.

    Args:

        theta (float) : The phase angle implemented on the qubit.

    """

    def __init__(self, theta: float) -> None:
        unitary = np.array(
            [
                [1, 0],
                [0, np.exp(1j * theta)],
            ]
        )
        super().__init__(unitary, f"P({round(theta, 3)})")


class Rx(Unitary):
    """
    Implements a Rx gate with angle theta across a pair of modes corresponding
    to a dual-rail encoded qubit.

    Args:

        theta (float) : The rotation angle of the gate.

    """

    def __init__(self, theta: float) -> None:
        unitary = np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )
        super().__init__(unitary, f"Rx({round(theta, 3)})")


class Ry(Unitary):
    """
    Implements a Ry gate with angle theta across a pair of modes corresponding
    to a dual-rail encoded qubit.

    Args:

        theta (float) : The rotation angle of the gate.

    """

    def __init__(self, theta: float) -> None:
        unitary = np.array(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )
        super().__init__(unitary, f"Ry({round(theta, 3)})")


class Rz(Unitary):
    """
    Implements a Rz gate with angle theta across a pair of modes corresponding
    to a dual-rail encoded qubit.

    Args:

        theta (float) : The rotation angle of the gate.

    """

    def __init__(self, theta: float) -> None:
        unitary = np.array(
            [
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)],
            ]
        )
        super().__init__(unitary, f"Rz({round(theta, 3)})")
