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
Contains a variety of three qubit components, designed for implementing required
qubit processing functionality in lightworks.
"""

import numpy as np

from lightworks.sdk.circuit import PhotonicCircuit, Unitary

from .single_qubit_gates import H


class CCZ(PhotonicCircuit):
    """
    Post-selected CCZ gate which acts across three dual-rail encoded qubits.
    There is a total of 4 heralded modes also included, each requiring 0
    photons on the input and output. For correct functioning of this gate it
    must be post-selected on the condition that only one photon is measured
    across the two modes used to encode each of the qubits.
    """

    def __init__(self) -> None:
        # Define transformation matrix
        # fmt: off
        u_a = np.array(
            [[3**-0.5, 0, 0, 1j * (2 / 3)**0.5, 0, 0, 0, 0, 0, 0],
             [0, 3**-0.5, 0, 0, 0, 1j * (2 / 3)**0.5, 0, 0, 0, 0],
             [0, 0, - 3**-0.5, 0, 0, 0, 0, 2**-0.5, -1j * 6**-0.5, 0],
             [-1j * (2 / 3)**0.5, 0, 0, -3**-0.5, 0, 0, 0, 0, 0, 0],
             [0, 0, -1j * 2**0.5 / 3, 0, -3**-0.5, 0, 0, -1j * 3**-0.5, 1 / 3,
              0],
             [0, -1j * (2 / 3)**0.5, 0, 0, 0, -3**-0.5, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1j * 2**-1.5, 0, 0, 1j * (7 / 8)**0.5],
             [0, 0, -1j * 3**-0.5, 0, 2**-0.5, 0, 0, -1j * 2**-1.5, -24**-0.5,
              0],
             [0, 0, -1j / 3, 0, -6**-0.5, 0, 0, 1j * 24**-0.5, -7 * 72**-0.5,
              0],
             [0, 0, 0, 0, 0, 0, -(7 / 8)**0.5, 0, 0, 2**-1.5]]
        )
        # fmt: on
        # Create unitary component and add heralds on required modes
        unitary = Unitary(u_a)
        unitary.herald(0, 0)
        unitary.herald(1, 0)
        unitary.herald(8, 0)
        unitary.herald(9, 0)

        super().__init__(6)
        self.add(unitary, 0, group=True, name="CCZ")


class CCNOT(PhotonicCircuit):
    """
    Post-selected CCNOT (Toffoli) gate which acts across three dual-rail
    encoded qubits. There is a total of 4 heralded modes also included, each
    requiring 0 photons on the input and output. For correct functioning of
    this gate it must be post-selected on the condition that only one photon is
    measured across the two modes used to encode each of the qubits.

    Args:

        target_qubit (int, optional) : Sets which of the three qubits is used as
            the target qubit for the gate. Should be either 0, 1 or 2.

    """

    def __init__(self, target_qubit: int = 2) -> None:
        if target_qubit not in {0, 1, 2}:
            raise ValueError(
                "target_qubit setting must have a value of either 0, 1 or 3."
            )

        # Create CCNOT from combination of H and CCZ
        circ = PhotonicCircuit(6)
        circ.add(H(), 2 * target_qubit)
        circ.add(CCZ(), 0)
        circ.add(H(), 2 * target_qubit)

        super().__init__(6)

        controls = tuple(i for i in range(3) if i != target_qubit)
        name = f"CCNOT ({controls}, {target_qubit})"
        self.add(circ, 0, group=True, name=name)
