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

from lightworks import State
from lightworks.emulator import Simulator
from lightworks.qubit import (H, X, Y, Z, S, T, CZ, CNOT, CZ_Heralded, 
                              CNOT_Heralded)

import pytest
import numpy as np

class TestSingleQubitGates:
    
    def test_hadamard(self):
        """Checks that the output from the Hadamard gate is correct."""
        sim = Simulator(H())
        # Input |1,0>
        results = sim.simulate(State([1,0]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 2**-0.5
        assert pytest.approx(results[State([0,1])], 1e-6) == 2**-0.5
        # Input |0,1>
        results = sim.simulate(State([0,1]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 2**-0.5
        assert pytest.approx(results[State([0,1])], 1e-6) == -2**-0.5
    
    def test_X(self):
        """Checks that the output from the X gate is correct."""
        sim = Simulator(X())
        # Input |1,0>
        results = sim.simulate(State([1,0]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 0
        assert pytest.approx(results[State([0,1])], 1e-6) == 1
        # Input |0,1>
        results = sim.simulate(State([0,1]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 1
        assert pytest.approx(results[State([0,1])], 1e-6) == 0
    
    def test_Y(self):
        """Checks that the output from the Y gate is correct."""
        sim = Simulator(Y())
        # Input |1,0>
        results = sim.simulate(State([1,0]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 0
        assert pytest.approx(results[State([0,1])], 1e-6) == 1j
        # Input |0,1>
        results = sim.simulate(State([0,1]))
        assert pytest.approx(results[State([1,0])], 1e-6) == -1j
        assert pytest.approx(results[State([0,1])], 1e-6) == 0
    
    def test_Z(self):
        """Checks that the output from the Z gate is correct."""
        sim = Simulator(Z())
        # Input |1,0>
        results = sim.simulate(State([1,0]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 1
        assert pytest.approx(results[State([0,1])], 1e-6) == 0
        # Input |0,1>
        results = sim.simulate(State([0,1]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 0
        assert pytest.approx(results[State([0,1])], 1e-6) == -1
    
    def test_S(self):
        """Checks that the output from the S gate is correct."""
        sim = Simulator(S())
        # Input |1,0>
        results = sim.simulate(State([1,0]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 1
        assert pytest.approx(results[State([0,1])], 1e-6) == 0
        # Input |0,1>
        results = sim.simulate(State([0,1]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 0
        assert pytest.approx(results[State([0,1])], 1e-6) == 1j
    
    def test_T(self):
        """Checks that the output from the T gate is correct."""
        sim = Simulator(T())
        # Input |1,0>
        results = sim.simulate(State([1,0]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 1
        assert pytest.approx(results[State([0,1])], 1e-6) == 0
        # Input |0,1>
        results = sim.simulate(State([0,1]))
        assert pytest.approx(results[State([1,0])], 1e-6) == 0
        assert pytest.approx(results[State([0,1])], 1e-6) == np.exp(1j*np.pi/4)

class TestTwoQubitGates:
    
    def test_CZ(self):
        """
        Checks that the output of the post-selected CZ gate is correct and that
        the success probability is 1/9.
        """
        # Define all input combinations
        states = [[1,0,1,0], [1,0,0,1], [0,1,1,0], [0,1,0,1]]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CZ())
        results = sim.simulate(states, states)
        # Check all results are identical except for |1,1> which has a -1
        amp = results[states[0], states[0]]
        assert pytest.approx(amp, 1e-6) == results[states[1], states[1]]
        assert pytest.approx(amp, 1e-6) == results[states[2], states[2]]
        assert pytest.approx(amp, 1e-6) == -results[states[3], states[3]]
        # Confirm success probability is 1/9
        assert pytest.approx(abs(amp)**2, 1e-6) == 1/9
    
    def test_CNOT(self):
        """
        Checks that the output of the post-selected CNOT gate is correct and 
        that the success probability is 1/9.
        """
        # Define all input combinations
        states = [[1,0,1,0], [1,0,0,1], [0,1,1,0], [0,1,0,1]]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CNOT())
        results = sim.simulate(states, states)
        # Check that swap occurs when control qubit is 1 but not otherwise
        amp = results[states[0], states[0]]
        assert pytest.approx(amp, 1e-6) == results[states[1], states[1]]
        assert pytest.approx(amp, 1e-6) == results[states[2], states[3]]
        assert pytest.approx(amp, 1e-6) == results[states[3], states[2]]
        # Confirm success probability is 1/9
        assert pytest.approx(abs(amp)**2, 1e-6) == 1/9
    
    def test_CZ_heralded(self):
        """
        Checks that the output of the heralded CZ gate is correct and that the 
        success probability is 1/16.
        """
        # Define all input combinations
        states = [[1,0,1,0], [1,0,0,1], [0,1,1,0], [0,1,0,1]]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CZ_Heralded())
        results = sim.simulate(states, states)
        # Check all results are identical except for |1,1> which has a -1
        amp = results[states[0], states[0]]
        assert pytest.approx(amp, 1e-6) == results[states[1], states[1]]
        assert pytest.approx(amp, 1e-6) == results[states[2], states[2]]
        assert pytest.approx(amp, 1e-6) == -results[states[3], states[3]]
        # Confirm success probability is 1/16
        assert pytest.approx(abs(amp)**2, 1e-6) == 1/16
    
    def test_CNOT_heralded(self):
        """
        Checks that the output of the heralded CNOT gate is correct and that 
        the success probability is 1/16.
        """
        # Define all input combinations
        states = [[1,0,1,0], [1,0,0,1], [0,1,1,0], [0,1,0,1]]
        states = [State(s) for s in states]
        # Calculate probability amplitudes
        sim = Simulator(CNOT_Heralded())
        results = sim.simulate(states, states)
        # Check that swap occurs when control qubit is 1 but not otherwise
        amp = results[states[0], states[0]]
        assert pytest.approx(amp, 1e-6) == results[states[1], states[1]]
        assert pytest.approx(amp, 1e-6) == results[states[2], states[3]]
        assert pytest.approx(amp, 1e-6) == results[states[3], states[2]]
        # Confirm success probability is 1/16
        assert pytest.approx(abs(amp)**2, 1e-6) == 1/16