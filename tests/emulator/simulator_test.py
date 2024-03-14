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

from lightworks import State, Unitary, random_unitary, Circuit, Parameter
from lightworks.emulator import PhotonNumberError
from lightworks.emulator import Simulator

import pytest

class TestSimulator:
    """
    Unit tests to check results returned from simulator in different cases,
    including when fermionic statistics are used.
    """
    
    def test_hom(self):
        """
        Checks the basic 2 mode hom case and confirms the probability of the
        |0,2> state is 0.5.
        """
        circ = Circuit(2)
        circ.add_bs(0)
        sim = Simulator(circ)
        results = sim.simulate(State([1,1]),State([2,0]))
        assert 0.5 == pytest.approx(abs(results.array[0,0])**2, 1e-8)
    
    def test_single_photon_case(self):
        """
        Runs a single photon sim and checks the calculated unitary matches the 
        target unitary.
        """
        N = 4
        U = random_unitary(N)
        unitary = Unitary(U)
        sim = Simulator(unitary)
        states = []
        for i in range(N):
            states.append(State([int(i==j) for j in range(N)]))
        results = sim.simulate(states, states)
        assert (U.T.round(8) == results.array.round(8)).all()
        
    def test_known_result(self):
        """
        Builds a circuit which produces a known result and checks this is found
        at the output.
        """
        # Build circuit
        circuit = Circuit(4)
        circuit.add_bs(1, reflectivity = 0.6)
        circuit.add_mode_swaps({0:1, 1:0, 2:3, 3:2})
        circuit.add_bs(0, 3, reflectivity = 0.3)
        circuit.add_bs(0)
        # And check output probability
        sim = Simulator(circuit)
        results = sim.simulate(State([1,0,0,1]), State([0,1,1,0]))
        assert abs(results.array[0,0])**2 == pytest.approx(0.5, 1e-8)
                
    def test_multi_photon_case(self):
        """
        Runs a multi-photon sim and checks the correct value is found for one 
        input/output.
        """
        U = random_unitary(4, seed = 10)
        unitary = Unitary(U)
        sim = Simulator(unitary)
        results = sim.simulate(State([1,0,1,0]), State([0,2,0,0]))
        x = results.array[0,0]
        assert x == pytest.approx(-0.18218877232689196-0.266230290128261j, 
                                  1e-8)
        
    def test_multi_photon_output_not_specified_case(self):
        """
        Runs a multi-photon sim and checks the correct value is found for one 
        input with outputs not specified.
        """
        U = random_unitary(4, seed = 10)
        unitary = Unitary(U)
        sim = Simulator(unitary)
        results = sim.simulate(State([1,0,1,0]))
        x = results[State([0,2,0,0])]
        assert x == pytest.approx(-0.18218877232689196-0.266230290128261j, 
                                  1e-8)
        
    def test_lossy_multi_photon_case(self):
        """
        Runs a lossy multi-photon sim and checks the correct value is found for
        one input/output.
        """
        circ = Circuit(4)
        circ.add_bs(0, loss = 2)
        circ.add_ps(1, phi = 0.3)
        circ.add_bs(1, loss = 2)
        circ.add_bs(2, loss = 2)
        circ.add_ps(1, phi = 0.5)
        circ.add_bs(1, loss = 2)
        sim = Simulator(circ)
        results = sim.simulate(State([2,0,0,0]), State([0,1,1,0]))
        x = results.array[0,0]
        assert x == pytest.approx(0.03647550871283556+0.01285838825922496j, 
                                  1e-8)
        
    def test_circuit_update(self):
        """Used to check circuit updates affect simulator results."""
        unitary = Unitary(random_unitary(4))
        # Create simulator and get initial results
        sim = Simulator(unitary)
        results = sim.simulate(State([1,0,1,0]), State([0,2,0,0]))
        x = results.array[0,0]
        # Update circuit adn re-simulate
        unitary.add_bs(0)
        results = sim.simulate(State([1,0,1,0]), State([0,2,0,0]))
        x2 = results.array[0,0]
        assert x != x2
        
    def test_circuit_parameter_update(self):
        """
        Used to check circuit updates through parameter changes affect 
        simulator results.
        """
        param = Parameter(0.3)
        circuit = Circuit(4)
        circuit.add_bs(0, reflectivity = param)
        circuit.add_bs(2, reflectivity = param)
        circuit.add_bs(1, reflectivity = param)
        # Create simulator and get initial results
        sim = Simulator(circuit)
        results = sim.simulate(State([1,0,1,0]), State([0,2,0,0]))
        x = results.array[0,0]
        # Update parameter and re-simulate
        param.set(0.65)
        results = sim.simulate(State([1,0,1,0]), State([0,2,0,0]))
        x2 = results.array[0,0]
        assert x != x2
        
    def test_circuit_assignment(self):
        """
        Checks that an incorrect value cannot be assigned to the circuit 
        attribute.
        """
        circuit = Unitary(random_unitary(4))
        sim = Simulator(circuit)
        with pytest.raises(TypeError):
            sim.circuit = random_unitary(5)
            
    def test_varied_input_n_raises_error(self):
        """
        Checks that an error is raised if it attempted to use inputs with 
        different photon numbers.
        """
        # Create circuit and simulator object
        circuit = Unitary(random_unitary(4))
        sim = Simulator(circuit)
        # Without output specified
        with pytest.raises(PhotonNumberError):
            sim.simulate([State([1,0,1,0]), State([0,1,0,0])])
        # With some outputs specified
        with pytest.raises(PhotonNumberError):
            sim.simulate([State([1,0,1,0]), State([0,1,0,0])],
                         [State([1,0,1,0]), State([0,1,0,1])])
            
    def test_varied_output_n_raises_error(self):
        """
        Checks that an error is raised if it attempted to use outputs with 
        different photon numbers to the input or each other.
        """
        # Create circuit and simulator object
        circuit = Unitary(random_unitary(4))
        sim = Simulator(circuit)
        # With different number to input
        with pytest.raises(PhotonNumberError):
            sim.simulate([State([1,0,1,0]), State([0,1,0,1])],
                         [State([0,0,1,0]), State([0,1,0,0])])
        # With different number to each other
        with pytest.raises(PhotonNumberError):
            sim.simulate([State([1,0,1,0]), State([0,1,0,1])],
                         [State([1,0,1,0]), State([0,1,0,0])])
