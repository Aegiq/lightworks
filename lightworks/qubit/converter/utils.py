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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


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
        if inst.operation.num_qubits >= 2 and inst.operation.name not in {
            "barrier",
            "swap",
        }:
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
