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

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate

from lightworks import State
from lightworks.emulator import Simulator
from lightworks.qubit.converter import qiskit_converter
from lightworks.qubit.converter.qiskit_convert import (
    SINGLE_QUBIT_GATES_MAP,
    THREE_QUBIT_GATES_MAP,
    TWO_QUBIT_GATES_MAP,
)


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

        sim = Simulator(conv_circ)
        results = sim.simulate(State([1, 0]), State([0, 1]))
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

        sim = Simulator(conv_circ)
        results = sim.simulate(State([0, 1, 1, 0]), State([0, 1, 0, 1]))
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

        sim = Simulator(conv_circ)
        results = sim.simulate(
            State([0, 1, 1, 0, 1, 0]), State([0, 1, 0, 1, 0, 1])
        )
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

        sim = Simulator(conv_circ)
        results = sim.simulate(State([0, 1, 1, 0]), State([0, 1, 0, 1]))
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

        sim = Simulator(conv_circ)
        results = sim.simulate(
            State([0, 1, 1, 0, 1, 0]), State([0, 1, 0, 1, 0, 1])
        )
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

        sim = Simulator(conv_circ)
        outputs = [State([1, 0, 1, 0]), State([0, 1, 0, 1])]
        results = sim.simulate(State([1, 0, 1, 0]), outputs)
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

        sim = Simulator(conv_circ)
        results = sim.simulate(State([1, 0, 0, 1]), State([0, 1, 0, 1]))
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

        sim = Simulator(conv_circ)
        results = sim.simulate(
            State([0, 1, 0, 1, 0, 1]), State([0, 1, 0, 1, 1, 0])
        )
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

        sim = Simulator(conv_circ)
        results = sim.simulate(
            State([0, 1, 0, 1, 0, 1]), State([1, 0, 0, 1, 0, 1])
        )
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

        sim = Simulator(conv_circ)
        results = sim.simulate(
            State([0, 1, 0, 1, 1, 0]), State([0, 1, 0, 1, 0, 1])
        )
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

        sim = Simulator(conv_circ)
        results = sim.simulate(
            State([0, 1, 0, 0, 2, 3]), State([2, 3, 0, 0, 0, 1])
        )
        assert abs(
            results[State([0, 1, 0, 0, 2, 3]), State([2, 3, 0, 0, 0, 1])]
        ) ** 2 == pytest.approx(1, 1e-6)
