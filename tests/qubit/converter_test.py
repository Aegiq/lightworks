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

import math
from random import choice, randint, random, sample

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate

from lightworks import Sampler, Simulator, State
from lightworks.emulator import Backend
from lightworks.qubit import qiskit_converter
from lightworks.qubit.converter.qiskit_convert import (
    ROTATION_GATES_MAP,
    SINGLE_QUBIT_GATES_MAP,
    THREE_QUBIT_GATES_MAP,
    TWO_QUBIT_GATES_MAP,
    convert_two_qubits_to_adjacent,
    post_selection_analyzer,
)

BACKEND = Backend("permanent")


class TestQiskitConversion:
    """
    Unit tests to check correct functionality of qiskit conversion function.
    """

    @pytest.mark.parametrize("gate", list(SINGLE_QUBIT_GATES_MAP.keys()))
    def test_all_single_qubit_gates(self, gate):
        """
        Checks all expected single qubit gates can be converted.
        """
        circ = QuantumCircuit(1)
        getattr(circ, gate)(0)
        qiskit_converter(circ)

    @pytest.mark.parametrize("gate", list(ROTATION_GATES_MAP.keys()))
    def test_all_rotation_qubit_gates(self, gate):
        """
        Checks all expected rotation qubit gates can be converted.
        """
        circ = QuantumCircuit(1)
        getattr(circ, gate)(math.tau * random(), 0)
        qiskit_converter(circ)

    @pytest.mark.parametrize("gate", list(TWO_QUBIT_GATES_MAP.keys()))
    def test_all_two_qubit_gates(self, gate):
        """
        Checks all expected two qubit gates can be converted.
        """
        circ = QuantumCircuit(2)
        getattr(circ, gate)(0, 1)
        qiskit_converter(circ)

    @pytest.mark.parametrize("gate", list(THREE_QUBIT_GATES_MAP.keys()))
    def test_all_three_qubit_gates(self, gate):
        """
        Checks all expected three qubit gates can be converted.
        """
        circ = QuantumCircuit(3)
        getattr(circ, gate)(0, 1, 2)
        qiskit_converter(circ, allow_post_selection=True)

    @pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
    def test_four_qubit_gate(self):
        """
        Checks that an error is raised for a 4 qubit gate.
        """
        circ = QuantumCircuit(4)
        circ.append(MCXGate(3), [0, 1, 2, 3])
        with pytest.raises(ValueError):
            qiskit_converter(circ)

    def test_x_gate(self):
        """
        Checks correct operation of a single qubit circuit with an X gate,
        flipping the qubit from the 0 to 1 state.
        """
        circ = QuantumCircuit(1)
        circ.x(0)
        conv_circ, _ = qiskit_converter(circ)

        sim = Simulator(conv_circ, State([1, 0]), State([0, 1]))
        results = BACKEND.run(sim)
        assert abs(results[State([1, 0]), State([0, 1])]) ** 2 == pytest.approx(
            1.0, 1e-6
        )

    def test_cnot(self):
        """
        Checks operation of two qubit CNOT gate produces output as expected.
        """
        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        conv_circ, _ = qiskit_converter(circ)

        sim = Simulator(conv_circ, State([0, 1, 1, 0]), State([0, 1, 0, 1]))
        results = BACKEND.run(sim)
        assert abs(
            results[State([0, 1, 1, 0]), State([0, 1, 0, 1])]
        ) ** 2 == pytest.approx(1 / 16, 1e-6)

    def test_cascaded_cnot(self):
        """
        Checks operation of two cascaded qubit CNOT gate produces output as
        expected.
        """
        circ = QuantumCircuit(3)
        circ.cx(0, 1)
        circ.cx(1, 2)
        conv_circ, _ = qiskit_converter(circ)

        sim = Simulator(
            conv_circ, State([0, 1, 1, 0, 1, 0]), State([0, 1, 0, 1, 0, 1])
        )
        results = BACKEND.run(sim)
        assert abs(
            results[State([0, 1, 1, 0, 1, 0]), State([0, 1, 0, 1, 0, 1])]
        ) ** 2 == pytest.approx(1 / 16**2, 1e-6)

    def test_cnot_post_selected(self):
        """
        Checks operation of two qubit post-selected CNOT gate produces output as
        expected.
        """
        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        conv_circ, _ = qiskit_converter(circ, allow_post_selection=True)

        sim = Simulator(conv_circ, State([0, 1, 1, 0]), State([0, 1, 0, 1]))
        results = BACKEND.run(sim)
        assert abs(
            results[State([0, 1, 1, 0]), State([0, 1, 0, 1])]
        ) ** 2 == pytest.approx(1 / 9, 1e-6)

    def test_cascaded_cnot_post_selected(self):
        """
        Checks operation of two cascaded qubit post-selected CNOT gate produces
        output as expected. Both gates should be post-selected, so the success
        probability should be 1/9*1/9 = 1/81.
        """
        circ = QuantumCircuit(3)
        circ.cx(0, 1)
        circ.cx(1, 2)
        conv_circ, _ = qiskit_converter(circ, allow_post_selection=True)

        sim = Simulator(
            conv_circ, State([0, 1, 1, 0, 1, 0]), State([0, 1, 0, 1, 0, 1])
        )
        results = BACKEND.run(sim)
        assert abs(
            results[State([0, 1, 1, 0, 1, 0]), State([0, 1, 0, 1, 0, 1])]
        ) ** 2 == pytest.approx(1 / 81, 1e-6)

    def test_bell_state(self):
        """
        Checks operation of two qubit H + CNOT gate produces the expected Bell
        state.
        """
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        conv_circ, _ = qiskit_converter(circ)

        outputs = [State([1, 0, 1, 0]), State([0, 1, 0, 1])]
        sim = Simulator(conv_circ, State([1, 0, 1, 0]), outputs)
        results = BACKEND.run(sim)
        for out in outputs:
            assert abs(results[State([1, 0, 1, 0]), out]) ** 2 == pytest.approx(
                1 / 32, 1e-6
            )

    def test_cnot_flipped(self):
        """
        Checks operation of two qubit CNOT gate produces output as expected when
        the control and the target qubits are flipped.
        """
        circ = QuantumCircuit(2)
        circ.cx(1, 0)
        conv_circ, _ = qiskit_converter(circ)

        sim = Simulator(conv_circ, State([1, 0, 0, 1]), State([0, 1, 0, 1]))
        results = BACKEND.run(sim)
        assert abs(
            results[State([1, 0, 0, 1]), State([0, 1, 0, 1])]
        ) ** 2 == pytest.approx(1 / 16, 1e-6)

    def test_cnot_non_adjacent(self):
        """
        Checks operation of CNOT gate when the two qubits it is applied are not
        adjacent to each other.
        """
        circ = QuantumCircuit(3)
        circ.cx(0, 2)
        conv_circ, _ = qiskit_converter(circ)

        sim = Simulator(
            conv_circ, State([0, 1, 0, 1, 0, 1]), State([0, 1, 0, 1, 1, 0])
        )
        results = BACKEND.run(sim)
        assert abs(
            results[State([0, 1, 0, 1, 0, 1]), State([0, 1, 0, 1, 1, 0])]
        ) ** 2 == pytest.approx(1 / 16, 1e-6)

    def test_cnot_non_adjacent_flipped(self):
        """
        Checks operation of CNOT gate when the two qubits it is applied are not
        adjacent to each other and the control and target are flipped.
        """
        circ = QuantumCircuit(3)
        circ.cx(2, 0)
        conv_circ, _ = qiskit_converter(circ)

        sim = Simulator(
            conv_circ, State([0, 1, 0, 1, 0, 1]), State([1, 0, 0, 1, 0, 1])
        )
        results = BACKEND.run(sim)
        assert abs(
            results[State([0, 1, 0, 1, 0, 1]), State([1, 0, 0, 1, 0, 1])]
        ) ** 2 == pytest.approx(1 / 16, 1e-6)

    def test_ccnot(self):
        """
        Checks operation of three qubit CCNOT gate produces output as expected.
        """
        circ = QuantumCircuit(3)
        circ.ccx(0, 1, 2)
        conv_circ, _ = qiskit_converter(circ, allow_post_selection=True)

        sim = Simulator(
            conv_circ, State([0, 1, 0, 1, 1, 0]), State([0, 1, 0, 1, 0, 1])
        )
        results = BACKEND.run(sim)
        assert abs(
            results[State([0, 1, 0, 1, 1, 0]), State([0, 1, 0, 1, 0, 1])]
        ) ** 2 == pytest.approx(1 / 72, 1e-6)

    def test_swap(self):
        """
        Checks swap is able to move correctly map between modes.
        """
        circ = QuantumCircuit(3)
        circ.swap(0, 2)
        conv_circ, _ = qiskit_converter(circ)

        sim = Simulator(
            conv_circ, State([0, 1, 0, 0, 2, 3]), State([2, 3, 0, 0, 0, 1])
        )
        results = BACKEND.run(sim)
        assert abs(
            results[State([0, 1, 0, 0, 2, 3]), State([2, 3, 0, 0, 0, 1])]
        ) ** 2 == pytest.approx(1, 1e-6)

    def test_complex_converter(self):
        """
        Tests conversion of a more complex qiskit circuit.
        """
        circ = QuantumCircuit(4)
        circ.x(0)
        circ.cx(0, 1)
        circ.y(0)
        circ.h(1)
        circ.cx(1, 2)
        circ.y(0)
        circ.z(2)
        circ.h(1)
        circ.cx(2, 3)
        qiskit_converter(circ, allow_post_selection=True)

    def test_post_selection(self):
        """
        Checks that post-selection object returned by the converter is able to
        create the required transformation.
        """
        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        conv_circ, post_select = qiskit_converter(
            circ, allow_post_selection=True
        )
        # Run sampler and check results
        n_samples = 10000
        sampler = Sampler(
            conv_circ,
            State([0, 1, 1, 0]),
            n_samples,
            post_selection=post_select,
        )
        results = BACKEND.run(sampler)
        assert results[State([0, 1, 0, 1])] == n_samples

    def test_post_selection_rules(self):
        """
        Checks that post-selection objected returned by the converter contains
        rules on the correct modes.
        """
        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        _, post_select = qiskit_converter(circ, allow_post_selection=True)

        r1_found = False
        r2_found = False
        for rule in post_select.rules:
            if rule.modes == (0, 1) and rule.n_photons == (1,):
                r1_found = True
            if rule.modes == (2, 3) and rule.n_photons == (1,):
                r2_found = True
        assert r1_found
        assert r2_found

    @pytest.mark.parametrize(
        ("q0", "q1"), [(0, 1), (1, 0), (2, 4), (4, 2), (2, 5), (5, 2)]
    )
    def test_convert_two_qubits_to_adjacent(self, q0, q1):
        """
        Checks that convert_two_qubits_to_adjacent function returns values with
        a difference of 1 and retains the correct order.
        """
        new_q0, new_q1, _ = convert_two_qubits_to_adjacent(q0, q1)
        assert abs(new_q1 - new_q0) == 1
        if q1 > q0:
            assert new_q1 > new_q0
        else:
            assert new_q1 < new_q0

    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_post_selection_analyzer(self, n_qubits):
        """
        Checks that post-selection analyzer returns the correct number of
        elements.
        """
        circuit = build_random_qiskit_circuit(n_qubits)
        post_selects, _ = post_selection_analyzer(circuit)
        assert len(post_selects) == len(circuit.data)


def build_random_qiskit_circuit(n_qubits):
    """
    Builds a random qiskit circuit for testing
    """
    if n_qubits < 2:
        raise ValueError("Number of qubits should be at least two.")
    gates = ["x", "y", "z", "cx", "cz"]
    if n_qubits >= 3:
        gates += ["ccx", "ccz"]
    circuit = QuantumCircuit(n_qubits)
    for _i in range(randint(10, 20)):
        gate = choice(gates)
        # Create unique list of qubits for each gate
        qubits = sample(range(n_qubits), len(gate))
        getattr(circuit, gate)(*qubits)
    return circuit
