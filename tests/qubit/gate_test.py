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

# ruff: noqa: N802, I001

import numpy as np
import pytest
from random import random

from lightworks import State, Simulator, Parameter
from lightworks.emulator import Backend

# fmt: off
from lightworks.qubit import (
    CCNOT, CCZ, CNOT, CZ, SWAP, CNOT_Heralded, CZ_Heralded, I, H, S, T, X, Y, Z,
    SX, Sadj, Tadj, P, Rx, Ry, Rz
)
import math
# fmt: on

BACKEND = Backend("permanent")


class TestSingleQubitGates:
    """
    Unit tests for all single qubit gates.
    """

    def test_I(self):
        """Checks that the output from the I gate is correct."""
        sim = Simulator(I(), State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 1
        assert pytest.approx(results[State([0, 1])], 1e-6) == 0
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 0
        assert pytest.approx(results[State([0, 1])], 1e-6) == 1

    def test_hadamard(self):
        """Checks that the output from the Hadamard gate is correct."""
        sim = Simulator(H(), State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 2**-0.5
        assert pytest.approx(results[State([0, 1])], 1e-6) == 2**-0.5
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 2**-0.5
        assert pytest.approx(results[State([0, 1])], 1e-6) == -(2**-0.5)

    def test_X(self):
        """Checks that the output from the X gate is correct."""
        sim = Simulator(X(), State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 0
        assert pytest.approx(results[State([0, 1])], 1e-6) == 1
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 1
        assert pytest.approx(results[State([0, 1])], 1e-6) == 0

    @pytest.mark.parametrize("theta", [math.pi, Parameter(math.pi)])
    def test_Rx(self, theta):
        """Checks that the output from the Rx gate is correct."""
        gate = Rx(theta)
        gate.ps(0, math.pi / 2)  # Make equivalent to X gate
        gate.ps(1, math.pi / 2)
        sim = Simulator(gate, State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 0
        assert pytest.approx(results[State([0, 1])], 1e-6) == 1
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 1
        assert pytest.approx(results[State([0, 1])], 1e-6) == 0

    def test_Y(self):
        """Checks that the output from the Y gate is correct."""
        sim = Simulator(Y(), State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 0
        assert pytest.approx(results[State([0, 1])], 1e-6) == 1j
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == -1j
        assert pytest.approx(results[State([0, 1])], 1e-6) == 0

    @pytest.mark.parametrize("theta", [math.pi, Parameter(math.pi)])
    def test_Ry(self, theta):
        """Checks that the output from the Ry gate is correct."""
        gate = Ry(theta)
        gate.ps(0, math.pi / 2)  # Make equivalent to Y gate
        gate.ps(1, math.pi / 2)
        sim = Simulator(gate, State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 0
        assert pytest.approx(results[State([0, 1])], 1e-6) == 1j
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == -1j
        assert pytest.approx(results[State([0, 1])], 1e-6) == 0

    def test_Z(self):
        """Checks that the output from the Z gate is correct."""
        sim = Simulator(Z(), State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 1
        assert pytest.approx(results[State([0, 1])], 1e-6) == 0
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 0
        assert pytest.approx(results[State([0, 1])], 1e-6) == -1

    @pytest.mark.parametrize("theta", [math.pi, Parameter(math.pi)])
    def test_Rz(self, theta):
        """Checks that the output from the Rz gate is correct."""
        gate = Rz(theta)
        gate.ps(0, math.pi / 2)  # Make equivalent to Z gate
        gate.ps(1, math.pi / 2)
        sim = Simulator(gate, State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 1
        assert pytest.approx(results[State([0, 1])], 1e-6) == 0
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 0
        assert pytest.approx(results[State([0, 1])], 1e-6) == -1

    def test_S(self):
        """Checks that the output from the S gate is correct."""
        sim = Simulator(S(), State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 1
        assert pytest.approx(results[State([0, 1])], 1e-6) == 0
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 0
        assert pytest.approx(results[State([0, 1])], 1e-6) == 1j

    def test_Sadj(self):
        """Checks that Sadj gate is hermitian conjugate of S."""
        assert pytest.approx(S().U, 1e-8) == np.conj(Sadj().U).T

    def test_T(self):
        """Checks that the output from the T gate is correct."""
        sim = Simulator(T(), State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 1
        assert pytest.approx(results[State([0, 1])], 1e-6) == 0
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 0
        assert pytest.approx(results[State([0, 1])], 1e-6) == np.exp(
            1j * np.pi / 4
        )

    def test_Tadj(self):
        """Checks that Tadj gate is hermitian conjugate of S."""
        assert pytest.approx(T().U, 1e-8) == np.conj(Tadj().U).T

    def test_SX(self):
        """Checks that the output from the SX gate is correct."""
        sim = Simulator(SX(), State([1, 0]))
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 1 / 2 * (1 + 1j)
        assert pytest.approx(results[State([0, 1])], 1e-6) == 1 / 2 * (1 - 1j)
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 1 / 2 * (1 - 1j)
        assert pytest.approx(results[State([0, 1])], 1e-6) == 1 / 2 * (1 + 1j)

    @pytest.mark.parametrize("phase", [None, Parameter(math.tau * random())])
    def test_P(self, phase):
        """Checks that the output from the phase gate is correct."""
        if phase is None:
            phase = math.tau * random()
        sim = Simulator(P(phase), State([1, 0]))
        if isinstance(phase, Parameter):
            phase = phase.get()
        # Input |1,0>
        results = BACKEND.run(sim)[State([1, 0])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 1
        assert pytest.approx(results[State([0, 1])], 1e-6) == 0
        # Input |0,1>
        sim.inputs = State([0, 1])
        results = BACKEND.run(sim)[State([0, 1])]
        assert pytest.approx(results[State([1, 0])], 1e-6) == 0
        assert pytest.approx(results[State([0, 1])], 1e-6) == np.exp(1j * phase)


class TestTwoQubitGates:
    """
    Unit tests for all single two gates.
    """

    def test_CZ(self):
        """
        Checks that the output of the post-selected CZ gate is correct and that
        the success probability is 1/9.
        """
        # Define all input combinations
        states = [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CZ(), states, states)
        results = BACKEND.run(sim)
        # Check all results are identical except for |1,1> which has a -1
        amp = results[states[0], states[0]]
        assert pytest.approx(amp, 1e-6) == results[states[1], states[1]]
        assert pytest.approx(amp, 1e-6) == results[states[2], states[2]]
        assert pytest.approx(amp, 1e-6) == -results[states[3], states[3]]
        # Confirm success probability is 1/9
        assert pytest.approx(abs(amp) ** 2, 1e-6) == 1 / 9

    def test_CNOT(self):
        """
        Checks that the output of the post-selected CNOT gate is correct and
        that the success probability is 1/9.
        """
        # Define all input combinations
        states = [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CNOT(), states, states)
        results = BACKEND.run(sim)
        # Check that swap occurs when control qubit is 1 but not otherwise
        amp = results[states[0], states[0]]
        assert pytest.approx(amp, 1e-6) == results[states[1], states[1]]
        assert pytest.approx(amp, 1e-6) == results[states[2], states[3]]
        assert pytest.approx(amp, 1e-6) == results[states[3], states[2]]
        # Confirm success probability is 1/9
        assert pytest.approx(abs(amp) ** 2, 1e-6) == 1 / 9

    def test_CNOT_flipped(self):
        """
        Checks that the output of the post-selected CNOT gate is correct and
        that the success probability is 1/9 when the target qubit is set to 0.
        """
        # Define all input combinations
        states = [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CNOT(target_qubit=0), states, states)
        results = BACKEND.run(sim)
        # Check that swap occurs when control qubit is 1 but not otherwise
        amp = results[states[0], states[0]]
        assert pytest.approx(amp, 1e-6) == results[states[2], states[2]]
        assert pytest.approx(amp, 1e-6) == results[states[1], states[3]]
        assert pytest.approx(amp, 1e-6) == results[states[3], states[1]]
        # Confirm success probability is 1/9
        assert pytest.approx(abs(amp) ** 2, 1e-6) == 1 / 9

    def test_CZ_heralded(self):
        """
        Checks that the output of the heralded CZ gate is correct and that the
        success probability is 1/16.
        """
        # Define all input combinations
        states = [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CZ_Heralded(), states, states)
        results = BACKEND.run(sim)
        # Check all results are identical except for |1,1> which has a -1
        amp = results[states[0], states[0]]
        assert pytest.approx(amp, 1e-6) == results[states[1], states[1]]
        assert pytest.approx(amp, 1e-6) == results[states[2], states[2]]
        assert pytest.approx(amp, 1e-6) == -results[states[3], states[3]]
        # Confirm success probability is 1/16
        assert pytest.approx(abs(amp) ** 2, 1e-6) == 1 / 16

    def test_CNOT_heralded(self):
        """
        Checks that the output of the heralded CNOT gate is correct and that
        the success probability is 1/16.
        """
        # Define all input combinations
        states = [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CNOT_Heralded(), states, states)
        results = BACKEND.run(sim)
        # Check that swap occurs when control qubit is 1 but not otherwise
        amp = results[states[0], states[0]]
        assert pytest.approx(amp, 1e-6) == results[states[1], states[1]]
        assert pytest.approx(amp, 1e-6) == results[states[2], states[3]]
        assert pytest.approx(amp, 1e-6) == results[states[3], states[2]]
        # Confirm success probability is 1/16
        assert pytest.approx(abs(amp) ** 2, 1e-6) == 1 / 16

    def test_CNOT_heralded_flipped(self):
        """
        Checks that the output of the heralded CNOT gate is correct and that
        the success probability is 1/16 when the target qubit is set to 0.
        """
        # Define all input combinations
        states = [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CNOT_Heralded(target_qubit=0), states, states)
        results = BACKEND.run(sim)
        # Check that swap occurs when control qubit is 1 but not otherwise
        amp = results[states[0], states[0]]
        assert pytest.approx(amp, 1e-6) == results[states[2], states[2]]
        assert pytest.approx(amp, 1e-6) == results[states[1], states[3]]
        assert pytest.approx(amp, 1e-6) == results[states[3], states[1]]
        # Confirm success probability is 1/16
        assert pytest.approx(abs(amp) ** 2, 1e-6) == 1 / 16

    def test_swap(self):
        """
        Checks that SWAP gate correctly maps modes to the correct location.
        """
        # Define gate
        swap = SWAP((0, 1), (4, 5))
        # Create simulator
        sim = Simulator(
            swap, State([1, 0, 0, 0, 0, 0]), State([0, 0, 0, 0, 1, 0])
        )
        # Check 0 mode of first qubit maps to 0 of second qubit
        results = BACKEND.run(sim)
        amp = results.array[0, 0]
        assert amp == pytest.approx(1 + 0j, 1e-6)
        # Check 1 mode of second qubit maps to 1 of first qubit
        sim.inputs = State([0, 0, 0, 0, 0, 1])
        sim.outputs = State([0, 1, 0, 0, 0, 0])
        results = BACKEND.run(sim)
        amp = results.array[0, 0]
        assert amp == pytest.approx(1 + 0j, 1e-6)


class TestThreeQubitGates:
    """
    Unit tests for all three qubit gates.
    """

    def test_CCZ(self):
        """
        Checks that the output of the post-selected CCZ gate is correct and
        that the success probability is 1/72.
        """
        # Define all input combinations
        states = [
            [1, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 1],
        ]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CCZ(), states, states)
        results = BACKEND.run(sim)
        # Check all results are identical except for |1,1,1> which has a -1
        amp = results[states[0], states[0]]
        for i in range(7):
            assert pytest.approx(amp, 1e-6) == results[states[i], states[i]]
        assert pytest.approx(amp, 1e-6) == -results[states[7], states[7]]
        # Confirm success probability is 1/72
        assert pytest.approx(abs(amp) ** 2, 1e-6) == 1 / 72

    @pytest.mark.parametrize(
        ("target", "swap_indices"), [(0, (5, 7)), (1, (6, 7)), (2, (3, 7))]
    )
    def test_CCNOT(self, target, swap_indices):
        """
        Checks that the output of the post-selected CCNOT gate is correct and
        that the success probability is 1/72. This is checked for all possible
        target qubits.
        """
        # Define all input combinations
        states = [
            [1, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 1],
        ]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CCNOT(target_qubit=target), states, states)
        results = BACKEND.run(sim)
        # Check swap occurs when both control qubits are 1 but not otherwise
        non_swapped = [i for i in range(8) if i not in swap_indices]
        amp = results[states[non_swapped[0]], states[non_swapped[0]]]
        for i in non_swapped:
            assert results[states[i], states[i]] == pytest.approx(amp, 1e-6)
        assert results[
            states[swap_indices[0]], states[swap_indices[1]]
        ] == pytest.approx(amp, 1e-6)
        assert results[
            states[swap_indices[1]], states[swap_indices[0]]
        ] == pytest.approx(amp, 1e-6)
        # Confirm success probability is 1/72
        assert pytest.approx(abs(amp) ** 2, 1e-6) == 1 / 72
