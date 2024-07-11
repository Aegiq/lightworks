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

from lightworks import State
from lightworks.emulator import Simulator
from lightworks.qubit.converter import qiskit_converter


class TestQiskitConversion:
    """
    Unit tests to check correct functionality of qiskit conversion function.
    """

    def test_x_gate(self):
        """
        Checks correct operation of a single qubit circuit with an X gate,
        flipping the qubit from the 0 to 1 state.
        """
        circ = QuantumCircuit(1)
        circ.x(0)
        conv_circ = qiskit_converter(circ)
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
        conv_circ = qiskit_converter(circ)
        sim = Simulator(conv_circ)
        results = sim.simulate(State([0, 1, 1, 0]), State([0, 1, 0, 1]))
        assert abs(
            results[State([0, 1, 1, 0]), State([0, 1, 0, 1])]
        ) ** 2 == pytest.approx(1 / 16, 1e-6)

    def test_cnot_flipped(self):
        """
        Checks operation of two qubit CNOT gate produces output as expected when
        the control and the target qubits are flipped.
        """
        circ = QuantumCircuit(2)
        circ.cx(1, 0)
        conv_circ = qiskit_converter(circ)
        sim = Simulator(conv_circ)
        results = sim.simulate(State([1, 0, 0, 1]), State([0, 1, 0, 1]))
        assert abs(
            results[State([1, 0, 0, 1]), State([0, 1, 0, 1])]
        ) ** 2 == pytest.approx(1 / 16, 1e-6)

    def test_ccnot(self):
        """
        Checks operation of three qubit CCNOT gate produces output as expected.
        """
        circ = QuantumCircuit(3)
        circ.ccx(0, 1, 2)
        conv_circ = qiskit_converter(circ)
        sim = Simulator(conv_circ)
        results = sim.simulate(
            State([0, 1, 0, 1, 1, 0]), State([0, 1, 0, 1, 0, 1])
        )
        assert abs(
            results[State([0, 1, 0, 1, 1, 0]), State([0, 1, 0, 1, 0, 1])]
        ) ** 2 == pytest.approx(1 / 72, 1e-6)
