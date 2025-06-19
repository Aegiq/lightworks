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
from lightworks.sdk.circuit import PhotonicCircuit
from lightworks.sdk.utils.exceptions import LightworksError
from lightworks.sdk.utils.post_selection import PostSelection

from . import QISKIT_INSTALLED

if QISKIT_INSTALLED:
    from qiskit import QuantumCircuit

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

ALLOWED_GATES = [
    *SINGLE_QUBIT_GATES_MAP,
    *ROTATION_GATES_MAP,
    *TWO_QUBIT_GATES_MAP,
    *THREE_QUBIT_GATES_MAP,
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
    converter = QiskitConverter(allow_post_selection)
    return converter.convert(circuit)


class QiskitConverter:
    """
    Manages conversion between qiskit and lightworks circuit, adding each of the
    qubit gates into a created circuit object.

    Args:

        allow_post_selection (bool, optional) : Controls whether post-selected
            gates can be utilised within the circuit.

    """

    def __init__(self, allow_post_selection: bool = False) -> None:
        self.allow_post_selection = allow_post_selection

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
        self.circuit = PhotonicCircuit(n_qubits * 2)
        self.modes = {i: (2 * i, 2 * i + 1) for i in range(n_qubits)}

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
            # Single Qubit Gates
            if len(qubits) == 1:
                if gate in SINGLE_QUBIT_GATES_MAP:
                    self._add_single_qubit_gate(gate, *qubits)
                else:
                    theta = inst.operation.params[0]
                    self._add_single_qubit_rotation_gate(gate, theta, *qubits)
            # Two Qubit Gates
            elif len(qubits) == 2:
                self._add_two_qubit_gate(
                    gate,
                    *qubits,  # type: ignore[call-arg]
                    post_select[i],
                )
            # Three Qubit Gates
            elif len(qubits) == 3:
                self._add_three_qubit_gate(
                    gate,
                    *qubits,  # type: ignore[call-arg]
                    post_select[i],
                )
            # Limit to three qubit gates
            else:
                raise ValueError("Gates with more than 3 qubits not supported.")

        if self.allow_post_selection:
            if ps_qubits:
                ps_rules = PostSelection()
                for q in ps_qubits:
                    ps_rules.add(self.modes[q], 1)
            else:
                ps_rules = None
        else:
            ps_rules = None

        return (self.circuit, ps_rules)

    def _add_single_qubit_gate(self, gate: str, qubit: int) -> None:
        """
        Adds a single qubit gate to the selected qubit on the circuit.
        """
        self.circuit.add(SINGLE_QUBIT_GATES_MAP[gate], self.modes[qubit][0])

    def _add_single_qubit_rotation_gate(
        self, gate: str, theta: float, qubit: int
    ) -> None:
        """
        Adds a single qubit gate to the selected qubit on the circuit.
        """
        self.circuit.add(ROTATION_GATES_MAP[gate](theta), self.modes[qubit][0])

    def _add_two_qubit_gate(
        self, gate: str, q0: int, q1: int, post_selection: bool = False
    ) -> None:
        """
        Adds a two qubit gate to the circuit on the selected qubits.
        """
        if gate == "swap":
            self.circuit.add(
                TWO_QUBIT_GATES_MAP["swap"](self.modes[q0], self.modes[q1]), 0
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
        self, gate: str, q0: int, q1: int, q2: int, post_selection: bool = False
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
                    "of the circuit as a result of requirements on "
                    "post-selection.."
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
            add_mode = self.modes[min(all_qubits)][0]
            self.circuit.add(add_circ, add_mode)
        else:
            msg = f"Unsupported gate '{gate}' included in circuit."
            raise ValueError(msg)


def convert_two_qubits_to_adjacent(
    q0: int, q1: int
) -> tuple[int, int, list[tuple[int, int]]]:
    """
    Takes two qubit indices and converts these so that they are adjacent to each
    other, and determining the swaps required for this. The order of the two
    qubits is preserved, so if q0 > q1 then this will remain True.

    Args:

        q0 (int) : First qubit which a gate acts on.

        q1 (int) : The second qubit which the gate acts on.

    Returns:

        int : The new first qubit which the gate should act on.

        int : The new second qubit which the gate should act on.

        list[tuple] : Pairs of qubits which swap gates should be applied to
            ensure the gate can act on the right qubits.

    """
    if abs(q1 - q0) == 1:
        return (q0, q1, [])
    swaps = []
    new_upper = max(q0, q1)
    new_lower = min(q0, q1)
    # Bring modes closer together until they are adjacent
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


def post_selection_analyzer(
    qc: "QuantumCircuit",
) -> tuple[list[bool], list[int]]:
    """
    Implements a basic algorithm to try to determine which gates can have
    post-selection and which require heralding. This is not necessarily optimal,
    but should at least reduce heralding requirements.

    Args:

        qc (QuantumCircuit) : The qiskit circuit to be analysed.

    Returns:

        list[bool] : A list of length elements in the circuit that indicates if
            each element is compatible with post-selection. This will include
            any single qubit gates, even though post-selection is not relevant
            here.

        list[int] : A list of integers indicating which qubits need a
            post-selection rule to be applied.

    """
    # First extract all qubit data from the circuit
    gate_qubits: list[list[int] | None] = []
    for inst in qc.data:
        if inst.operation.num_qubits >= 2:
            gate_qubits.append(
                [
                    inst.qubits[i]._index
                    for i in range(inst.operation.num_qubits)
                ]
            )
        else:
            gate_qubits.append(None)

    post_selection = []
    has_ps = []
    # Work backwards through gates
    for gate in reversed(gate_qubits):
        if gate is None:
            post_selection.append(False)
            continue
        can_ps = not all(q in has_ps for q in gate)
        post_selection.append(can_ps)
        has_ps += gate
    # Return if a gate can have post-selection and all modes which will require
    # it.
    return list(reversed(post_selection)), list(set(has_ps))
