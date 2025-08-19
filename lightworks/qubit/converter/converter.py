# Copyright 2025 - 2025 Aegiq Ltd.
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

import numpy as np

from lightworks.qubit.gates.single_qubit_gates import (
    SX,
    H,
    P,
    Rx,
    Ry,
    Rz,
    S,
    Sadj,
    T,
    Tadj,
    X,
    Y,
    Z,
)
from lightworks.qubit.gates.three_qubit_gates import CCNOT, CCZ
from lightworks.qubit.gates.two_qubit_gates import (
    CNOT,
    CZ,
    SWAP,
    CNOT_Heralded,
    CZ_Heralded,
)
from lightworks.sdk.circuit import PhotonicCircuit, Unitary

from .utils import convert_two_qubits_to_adjacent

SINGLE_QUBIT_GATES_MAP = {
    "h": H(),
    "x": X(),
    "y": Y(),
    "z": Z(),
    "s": S(),
    "sdg": Sadj(),
    "t": T(),
    "tdg": Tadj(),
    "sx": SX(),
}
ROTATION_GATES_MAP = {"rx": Rx, "ry": Ry, "rz": Rz, "p": P}

TWO_QUBIT_GATES_MAP = {"cx": CNOT_Heralded, "cz": CZ_Heralded, "swap": SWAP}
TWO_QUBIT_GATES_MAP_PS = {"cx": CNOT, "cz": CZ}

THREE_QUBIT_GATES_MAP = {"ccx": CCNOT, "ccz": CCZ}


class Converter:
    """
    TODO: DESC
    """

    def __init__(
        self, n_qubits: int, allow_post_selection: bool = False
    ) -> None:
        self.allow_post_selection = allow_post_selection
        self._modes = {i: (2 * i, 2 * i + 1) for i in range(n_qubits)}

    def _add_single_qubit_gate(
        self, circuit: PhotonicCircuit, gate: str, qubit: int
    ) -> None:
        """
        Adds a single qubit gate to the selected qubit on the circuit.
        """
        circuit.add(SINGLE_QUBIT_GATES_MAP[gate], self._modes[qubit][0])

    def _add_single_qubit_rotation_gate(
        self, circuit: PhotonicCircuit, gate: str, theta: float, qubit: int
    ) -> None:
        """
        Adds a single qubit gate to the selected qubit on the circuit.
        """
        circuit.add(ROTATION_GATES_MAP[gate](theta), self._modes[qubit][0])

    def _add_single_qubit_unitary(
        self, circuit: PhotonicCircuit, params: list[float], qubit: int
    ) -> None:
        """
        Adds an arbitrary single qubit rotation unitary to a circuit
        """
        if len(params) != 3:
            raise ValueError("Expected unitary gate to have 3 parameters.")
        theta, phi, lam = params
        unitary = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lam)) * np.cos(theta / 2),
                ],
            ]
        )
        circuit.add(Unitary(unitary), self._modes[qubit][0])

    def _add_two_qubit_gate(
        self,
        circuit: PhotonicCircuit,
        gate: str,
        q0: int,
        q1: int,
        post_selection: bool = False,
    ) -> None:
        """
        Adds a two qubit gate to the circuit on the selected qubits.
        """
        if gate == "swap":
            circuit.add(
                TWO_QUBIT_GATES_MAP["swap"](self._modes[q0], self._modes[q1]), 0
            )
        elif gate in {"cx", "cz"}:
            mapper = (
                TWO_QUBIT_GATES_MAP
                if not post_selection
                else TWO_QUBIT_GATES_MAP_PS
            )
            q0, q1, to_swap = convert_two_qubits_to_adjacent(q0, q1)
            if gate == "cx":
                target = q1 - min([q0, q1])
                add_circ = mapper["cx"](target)
            else:
                add_circ = mapper["cz"]()
            add_mode = self._modes[min([q0, q1])][0]
            for swap_qs in to_swap:
                self._add_two_qubit_gate(
                    circuit, "swap", swap_qs[0], swap_qs[1]
                )
            circuit.add(add_circ, add_mode)
            for swap_qs in to_swap:
                self._add_two_qubit_gate(
                    circuit, "swap", swap_qs[0], swap_qs[1]
                )
        else:
            msg = f"Unsupported gate '{gate}' included in circuit."
            raise ValueError(msg)

    def _add_three_qubit_gate(
        self,
        circuit: PhotonicCircuit,
        gate: str,
        q0: int,
        q1: int,
        q2: int,
        post_selection: bool = False,
    ) -> None:
        """
        Adds a three qubit gate to the circuit on the selected qubits.
        """
        if gate in {"ccx", "ccz"}:
            if not post_selection:
                raise ValueError(
                    "Three qubit gates can only be used with post-selection. "
                    "Ensure allow_post_selection is True to enable this. The "
                    "location of the gate may also need to be towards the end "
                    "of the circuit as a result of the requirements on "
                    "post-selection."
                )
            all_qubits = [q0, q1, q2]
            if max(all_qubits) - min(all_qubits) != 2:
                raise ValueError(
                    "CCX and CCZ qubits must be adjacent to each other, "
                    "please add swap gates to achieve this."
                )
            if gate == "ccx":
                target = q2 - min(all_qubits)
                add_circ = THREE_QUBIT_GATES_MAP["ccx"](target)
            else:
                add_circ = THREE_QUBIT_GATES_MAP["ccz"]()
            add_mode = self._modes[min(all_qubits)][0]
            circuit.add(add_circ, add_mode)
        else:
            msg = f"Unsupported gate '{gate}' included in circuit."
            raise ValueError(msg)

    def _add_barrier(self, circuit: PhotonicCircuit, qubits: list[int]) -> None:
        """
        Adds a barrier to the circuit on the provided qubits.
        """
        modes = [self._modes[q][i] for q in qubits for i in range(2)]
        circuit.barrier(modes)
