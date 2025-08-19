# Copyright 2024 - 2025 Aegiq Ltd.
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

from lightworks.sdk.circuit import PhotonicCircuit
from lightworks.sdk.utils.exceptions import LightworksError
from lightworks.sdk.utils.post_selection import PostSelection

from . import QISKIT_INSTALLED
from .converter import (
    ROTATION_GATES_MAP,
    SINGLE_QUBIT_GATES_MAP,
    THREE_QUBIT_GATES_MAP,
    TWO_QUBIT_GATES_MAP,
    Converter,
)
from .utils import post_selection_analyzer

if QISKIT_INSTALLED:
    from qiskit import QuantumCircuit


ALLOWED_GATES = [
    *SINGLE_QUBIT_GATES_MAP,
    *ROTATION_GATES_MAP,
    *TWO_QUBIT_GATES_MAP,
    *THREE_QUBIT_GATES_MAP,
    "barrier",
    "u",
]


def qiskit_converter(
    circuit: "QuantumCircuit", allow_post_selection: bool = False
) -> tuple[PhotonicCircuit, PostSelection | None]:
    """
    Performs conversion of a provided qiskit QuantumCircuit into a photonic
    circuit within Lightworks.

    Args:

        circuit (QuantumCircuit) : The qiskit circuit to be converted.

        allow_post_selection (bool, optional) : Controls whether post-selected
            gates can be utilised within the circuit.

    Returns:

        PhotonicCircuit : The created circuit within Lightworks.

        PostSelection | None : If post-selection rules are required for the
            created circuit, then an object which implements these will be
            returned, otherwise it will be None.

    """
    converter = QiskitConverter(circuit.num_qubits, allow_post_selection)
    return converter.convert(circuit)


class QiskitConverter(Converter):
    """
    Manages conversion between qiskit and lightworks circuit, adding each of the
    qubit gates into a created circuit object.

    Args:

        allow_post_selection (bool, optional) : Controls whether post-selected
            gates can be utilised within the circuit.

    """

    def __init__(
        self, n_qubits: int, allow_post_selection: bool = False
    ) -> None:
        super().__init__(n_qubits, allow_post_selection)

    def convert(
        self, q_circuit: "QuantumCircuit"
    ) -> tuple[PhotonicCircuit, PostSelection | None]:
        """
        Performs conversion of a provided qiskit QuantumCircuit into a photonic
        circuit within Lightworks.

        Args:

            q_circuit (QuantumCircuit) : The qiskit circuit to be converted.

        Returns:

            PhotonicCircuit : The created circuit within Lightworks.

            PostSelection | None : If post-selection rules are required for the
                created circuit, then an object which implements these will be
                returned, otherwise it will be None.

        """
        if not QISKIT_INSTALLED:
            raise LightworksError(
                "Lightworks qiskit optional requirements not installed, "
                "this can be achieved with 'pip install lightworks[qiskit]'."
            )

        if not isinstance(q_circuit, QuantumCircuit):
            raise TypeError(
                "PhotonicCircuit to convert must be a qiskit circuit."
            )

        n_qubits = q_circuit.num_qubits
        circuit = PhotonicCircuit(n_qubits * 2)

        if self.allow_post_selection:
            post_select, ps_qubits = post_selection_analyzer(q_circuit)
        else:
            post_select = [False] * len(q_circuit.data)

        for i, inst in enumerate(q_circuit.data):
            gate = inst.operation.name
            qubits = [
                inst.qubits[i]._index for i in range(inst.operation.num_qubits)
            ]
            if gate not in ALLOWED_GATES:
                msg = f"Unsupported gate '{gate}' included in circuit."
                raise ValueError(msg)
            # First catch barriers
            if gate == "barrier":
                self._add_barrier(circuit, qubits)
            # Single Qubit Gates
            elif len(qubits) == 1:
                if gate == "u":
                    self._add_single_qubit_unitary(
                        circuit, inst.params, qubits[0]
                    )
                elif gate in SINGLE_QUBIT_GATES_MAP:
                    self._add_single_qubit_gate(circuit, gate, qubits[0])
                else:
                    theta = inst.operation.params[0]
                    self._add_single_qubit_rotation_gate(
                        circuit, gate, theta, qubits[0]
                    )
            # Two Qubit Gates
            elif len(qubits) == 2:
                self._add_two_qubit_gate(
                    circuit,
                    gate,
                    qubits[0],
                    qubits[1],
                    post_select[i],
                )
            # Three Qubit Gates
            elif len(qubits) == 3:
                self._add_three_qubit_gate(
                    circuit,
                    gate,
                    qubits[0],
                    qubits[1],
                    qubits[2],
                    post_select[i],
                )
            # Limit to three qubit gates
            else:
                raise ValueError("Gates with more than 3 qubits not supported.")

        ps_rules = None
        if self.allow_post_selection and ps_qubits:
            ps_rules = PostSelection()
            for q in ps_qubits:
                ps_rules.add(self._modes[q], 1)

        return (circuit, ps_rules)
