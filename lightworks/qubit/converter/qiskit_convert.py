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

TWO_QUBIT_GATES_MAP = {
    "cx": CNOT_Heralded(),
    "cz": CZ_Heralded(),
}

THREE_QUBIT_GATES_MAP = {
    "ccx": CCNOT(),
    "cxz": CCZ(),
}

ALLOWED_GATES = [
    *SINGLE_QUBIT_GATES_MAP.keys(),
    *TWO_QUBIT_GATES_MAP.keys(),
    "swap",
]


def qiskit_converter(q_circ: QuantumCircuit) -> Circuit:
    """
    Desc
    """
    if not isinstance(q_circ, QuantumCircuit):
        raise TypeError("Circuit to convert must be a qiskit circuit.")

    n_qubits = q_circ.num_qubits
    circuit = Circuit(n_qubits * 2)
    modes = {i: (2 * i, 2 * i + 1) for i in range(n_qubits)}

    for inst in q_circ.data:
        gate = inst.operation.name

        # Single Qubit Gates
        if inst.operation.num_qubits == 1:
            qubit = inst.qubits[0]._index
            if gate not in ALLOWED_GATES:
                msg = f"Unsupported gate '{gate}' included in circuit."
                raise ValueError(msg)
            circuit.add(SINGLE_QUBIT_GATES_MAP[gate], modes[qubit][0])

        # Two Qubit Gates
        elif inst.operation.num_qubits == 2:
            q0 = inst.qubits[0]._index
            q1 = inst.qubits[1]._index
            if gate == "swap":
                circuit.add(SWAP(modes[q0], modes[q1]), 0)
            elif gate in ["cx", "cz"]:
                if q1 - q0 != 1:
                    raise ValueError(
                        "CX and CZ must have control qubit q and target qubit "
                        "q+1, please add swap gates to achieve this."
                    )
                circuit.add(TWO_QUBIT_GATES_MAP[gate], modes[q0][0])
            else:
                msg = f"Unsupported gate '{gate}' included in circuit."
                raise ValueError(msg)

        # Three Qubit Gates
        elif inst.operation.num_qubits == 3:
            q0 = inst.qubits[0]._index
            q1 = inst.qubits[1]._index
            q2 = inst.qubits[2]._index
            if gate in ["ccx", "ccz"]:
                if q1 < q0:
                    q0, q1 = q1, q0
                if q2 - q1 != 1 or q2 - q0 != 2:
                    raise ValueError(
                        "CCX and CCZ must have control qubit q & q + 1 and "
                        "target qubit q+2, please add swap gates to achieve "
                        "this."
                    )
                circuit.add(THREE_QUBIT_GATES_MAP[gate], modes[q0][0])
            else:
                msg = f"Unsupported gate '{gate}' included in circuit."
                raise ValueError(msg)

        # Limit to three qubit gates
        else:
            raise ValueError("Gates with more than 3 qubits not supported.")

    return circuit
