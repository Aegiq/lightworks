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

from qiskit import QuantumCircuit

from ...sdk.circuit import Circuit
from ..gates import (
    CCNOT,
    CCZ,
    SWAP,
    CNOT_Heralded,
    CZ_Heralded,
    H,
    S,
    T,
    X,
    Y,
    Z,
)

SINGLE_QUBIT_GATES_MAP = {
    "h": H(),
    "x": X(),
    "y": Y(),
    "z": Z(),
    "s": S(),
    "t": T(),
}

TWO_QUBIT_GATES_MAP = {"cx": CNOT_Heralded, "cz": CZ_Heralded, "swap": SWAP}

THREE_QUBIT_GATES_MAP = {"ccx": CCNOT, "ccz": CCZ}

ALLOWED_GATES = [
    *SINGLE_QUBIT_GATES_MAP,
    *TWO_QUBIT_GATES_MAP,
    *THREE_QUBIT_GATES_MAP,
]


def qiskit_converter(circuit: QuantumCircuit) -> Circuit:
    """
    Performs conversion of a provided qiskit QuantumCircuit into a photonic
    circuit within Lightworks.

    Args:

        circuit (QuantumCircuit) : The qiskit circuit to be converted,

    Returns:

        Circuit : The created circuit within Lightworks.

    """
    converter = QiskitConverter()
    return converter.convert(circuit)


class QiskitConverter:
    """
    Manages conversion between qiskit and lightworks circuit, adding each of the
    qubit gates into a created circuit object.
    """

    def convert(self, q_circuit: QuantumCircuit) -> Circuit:
        """
        Performs conversion of a provided qiskit QuantumCircuit into a photonic
        circuit within Lightworks.

        Args:

            q_circuit (QuantumCircuit) : The qiskit circuit to be converted,

        Returns:

            Circuit : The created circuit within Lightworks.

        """
        if not isinstance(q_circuit, QuantumCircuit):
            raise TypeError("Circuit to convert must be a qiskit circuit.")

        n_qubits = q_circuit.num_qubits
        self.circuit = Circuit(n_qubits * 2)
        self.modes = {i: (2 * i, 2 * i + 1) for i in range(n_qubits)}

        has_three_qubit = False
        for inst in q_circuit.data:
            gate = inst.operation.name
            qubits = [
                inst.qubits[i]._index for i in range(inst.operation.num_qubits)
            ]
            if gate not in ALLOWED_GATES:
                msg = f"Unsupported gate '{gate}' included in circuit."
                raise ValueError(msg)
            three_qubit_msg = (
                "When a three qubit gate is included then this must be the only"
                " multi-qubit gate in the circuit."
            )
            # Single Qubit Gates
            if len(qubits) == 1:
                self._add_single_qubit_gate(gate, *qubits)
            # Two Qubit Gates
            elif len(qubits) == 2:
                if has_three_qubit:
                    raise ValueError(three_qubit_msg)
                self._add_two_qubit_gate(gate, *qubits)
            # Three Qubit Gates
            elif len(qubits) == 3:
                if has_three_qubit:
                    raise ValueError(three_qubit_msg)
                has_three_qubit = True
                self._add_three_qubit_gate(gate, *qubits)
            # Limit to three qubit gates
            else:
                raise ValueError("Gates with more than 3 qubits not supported.")

        return self.circuit

    def _add_single_qubit_gate(self, gate: str, qubit: int) -> None:
        """
        Adds a single qubit gate to the provided qubit on the circuit.
        """
        self.circuit.add(SINGLE_QUBIT_GATES_MAP[gate], self.modes[qubit][0])

    def _add_two_qubit_gate(self, gate: str, q0: int, q1: int) -> None:
        """
        Adds a provided two qubit gate within an instruction to a circuit on
        the correct modes.
        """
        if gate == "swap":
            self.circuit.add(
                TWO_QUBIT_GATES_MAP["swap"](self.modes[q0], self.modes[q1]), 0
            )
        elif gate in ["cx", "cz"]:
            q0, q1, to_swap = convert_two_qubits_to_adjacent(q0, q1)
            if gate == "cx":
                target = q1 - min([q0, q1])
                add_circ = TWO_QUBIT_GATES_MAP["cx"](target)
            else:
                add_circ = TWO_QUBIT_GATES_MAP["cz"]()
            add_mode = self.modes[min([q0, q1])][0]
            for swap_qs in to_swap:
                self._add_two_qubit_gate("swap", swap_qs[0], swap_qs[1])
            self.circuit.add(add_circ, add_mode)
            for swap_qs in to_swap:
                self._add_two_qubit_gate("swap", swap_qs[0], swap_qs[1])
        else:
            msg = f"Unsupported gate '{gate}' included in circuit."
            raise ValueError(msg)

    def _add_three_qubit_gate(
        self, gate: str, q0: int, q1: int, q2: int
    ) -> None:
        """
        Adds a provided three qubit gate within an instruction to a circuit on
        the correct modes.
        """
        if gate in ["ccx", "ccz"]:
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
            add_mode = self.modes[min(all_qubits)][0]
            self.circuit.add(add_circ, add_mode)
        else:
            msg = f"Unsupported gate '{gate}' included in circuit."
            raise ValueError(msg)


def convert_two_qubits_to_adjacent(q0: int, q1: int) -> tuple[int, int, list]:
    """
    Takes two qubit indices and converts these so that they are adjacent to each
    other, and determining the swaps required for this. The order of the two
    qubits is preserved, so if q0 > q1 then this will remain True.
    """
    if abs(q1 - q0) == 1:
        return (q0, q1, [])
    swaps = []
    new_upper = max(q0, q1)
    new_lower = min(q0, q1)
    while new_upper - new_lower != 1:
        new_upper -= 1
        if new_upper - new_lower == 1:
            break
        new_lower += 1
    if min(q0, q1) != new_lower:
        swaps.append((min(q0, q1), new_lower))
    if max(q0, q1) != new_upper:
        swaps.append((max(q0, q1), new_upper))
    if q0 < q1:
        q0, q1 = new_lower, new_upper
    else:
        q0, q1 = new_upper, new_lower
    return (q0, q1, swaps)
