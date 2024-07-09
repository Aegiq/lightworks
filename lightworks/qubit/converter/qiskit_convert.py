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
from qiskit.circuit import CircuitInstruction

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

TWO_QUBIT_GATES_MAP = {
    "cx": CNOT_Heralded,
    "cz": CZ_Heralded,
}

THREE_QUBIT_GATES_MAP = {
    "ccx": CCNOT,
    "cxz": CCZ,
}

ALLOWED_GATES = [
    *SINGLE_QUBIT_GATES_MAP,
    *TWO_QUBIT_GATES_MAP,
    "swap",
]

# TODO: Track post-selection requirements, using post-selected gates where
# possible and return required post-selection.
# TODO: Support cx and cz on non-adjacent qubits.


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
    Desc
    """

    def convert(self, q_circ: QuantumCircuit) -> Circuit:
        """
        Performs conversion of a provided qiskit QuantumCircuit into a photonic
        circuit within Lightworks.

        Args:

            q_circ (QuantumCircuit) : The qiskit circuit to be converted,

        Returns:

            Circuit : The created circuit within Lightworks.

        """
        if not isinstance(q_circ, QuantumCircuit):
            raise TypeError("Circuit to convert must be a qiskit circuit.")

        n_qubits = q_circ.num_qubits
        self.circuit = Circuit(n_qubits * 2)
        self.modes = {i: (2 * i, 2 * i + 1) for i in range(n_qubits)}

        for inst in q_circ.data:
            # Single Qubit Gates
            if inst.operation.num_qubits == 1:
                self.add_single_qubit_gate(inst)

            # Two Qubit Gates
            elif inst.operation.num_qubits == 2:
                self.add_two_qubit_gate(inst)

            # Three Qubit Gates
            elif inst.operation.num_qubits == 3:
                self.add_three_qubit_gate(inst)

            # Limit to three qubit gates
            else:
                raise ValueError("Gates with more than 3 qubits not supported.")

        return self.circuit

    def add_single_qubit_gate(self, instruction: CircuitInstruction) -> None:
        """
        Desc
        """
        gate = instruction.operation.name
        qubit = instruction.qubits[0]._index
        if gate not in ALLOWED_GATES:
            msg = f"Unsupported gate '{gate}' included in circuit."
            raise ValueError(msg)
        self.circuit.add(SINGLE_QUBIT_GATES_MAP[gate], self.modes[qubit][0])

    def add_two_qubit_gate(self, instruction: CircuitInstruction) -> None:
        """
        Desc
        """
        gate = instruction.operation.name
        q0 = instruction.qubits[0]._index
        q1 = instruction.qubits[1]._index
        if gate == "swap":
            self.circuit.add(SWAP(self.modes[q0], self.modes[q1]), 0)
        elif gate in ["cx", "cz"]:
            if abs(q1 - q0) != 1:
                raise ValueError(
                    "CX and CZ qubits must be adjacent to each other, "
                    "please add swap gates to achieve this."
                )
            if gate == "cx":
                target = q1 - min([q0, q1])
                add_circ = TWO_QUBIT_GATES_MAP["cx"](target)
            else:
                add_circ = TWO_QUBIT_GATES_MAP["cz"]()
            add_mode = self.modes[min([q0, q1])][0]
            self.circuit.add(add_circ, add_mode)
        else:
            msg = f"Unsupported gate '{gate}' included in circuit."
            raise ValueError(msg)

    def add_three_qubit_gate(self, instruction: CircuitInstruction) -> None:
        """
        Desc
        """
        gate = instruction.operation.name
        q0 = instruction.qubits[0]._index
        q1 = instruction.qubits[1]._index
        q2 = instruction.qubits[2]._index
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
