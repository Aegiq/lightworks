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

from lightworks import State, Unitary, Circuit, random_unitary, Parameter
from lightworks.emulator import QuickSampler, Sampler, StateError
from lightworks.emulator import set_statistic_type
import unittest

class QuickSamplerTest(unittest.TestCase):
    """
    Unit tests to check results produced by QuickSampler object in the 
    emulator.
    """
    
    def test_hom(self):
        """
        Checks sampling a basic 2 photon input onto a 50:50 beam splitter, 
        which should undergo HOM, producing outputs of |2,0> and |0,2>.
        """
        circuit = Circuit(2)
        circuit.add_bs(0)
        sampler = QuickSampler(circuit, State([1,1]))
        N_sample = 100000
        results = sampler.sample_N_outputs(N_sample, seed = 21)
        self.assertEqual(len(results), 2)
        self.assertTrue(0.49 < results[State([2,0])]/N_sample < 0.51)
        self.assertTrue(0.49 < results[State([0,2])]/N_sample < 0.51)
        
    def test_equivalence(self):
        """
        Confirms that the Sampler and QuickSampler produce identical results
        in situations where the QuickSampler assumptions hold true.
        """
        circuit = Unitary(random_unitary(4))
        sampler = Sampler(circuit, State([1,0,1,0]))
        p1 = sampler.probability_distribution
        q_sampler = QuickSampler(circuit, State([1,0,1,0]))
        p2 = q_sampler.probability_distribution
        # Loop through distributions and check they are equal to a reasonable
        # accuracy
        for s1 in p1:
            if s1 not in p2: # Cannot be identical if a state is missed
                self.fail("Missing state in QuickSampler distribution.")
            if round(p1[s1], 8) != round(p2[s1], 8): # Checks equivalence
                self.fail("Probabilities not equivalent.")
    
    def test_known_result(self):
        """
        Builds a circuit which produces a known result and checks this is found
        at the output.
        """
        # Build circuit
        circuit = Circuit(4)
        circuit.add_bs(1)
        circuit.add_mode_swaps({0:1, 1:0, 2:3, 3:2})
        circuit.add_bs(0, 3)
        # And check output counts
        sampler = QuickSampler(circuit, State([1,0,0,1]))
        results = sampler.sample_N_outputs(1000)
        self.assertEqual(results[State([0,1,1,0])], 1000)
        
    def test_sampling(self):
        """
        Checks that the probability distribution calculated by the sampler is 
        correct.
        """
        unitary = Unitary(random_unitary(4, seed = 43))
        sampler = QuickSampler(unitary, State([1,0,1,0]), 
                               photon_counting = False, 
                               herald = lambda s: s[0] == 0)
        p = sampler.probability_distribution[State([0,1,1,0])]
        self.assertAlmostEqual(p, 0.3156177858, 8)
        
    def test_sampling_2photons_in_mode(self):
        """
        Checks that the probability distribution calculated by the sampler is 
        correct when using 2 photons in a single mode.
        """
        unitary = Unitary(random_unitary(4, seed = 43))
        sampler = QuickSampler(unitary, State([0,2,0,0]), 
                               photon_counting = False, 
                               herald = lambda s: s[0] == 0)
        p = sampler.probability_distribution[State([0,1,1,0])]
        self.assertAlmostEqual(p, 0.071330233065, 8)
            
    def test_lossy_sampling(self):
        """
        Checks that the probability distribution calculated by the sampler with
        a lossy circuit is correct.
        """
        circuit = Circuit(4)
        circuit.add_bs(0, loss = 1.3)
        circuit.add_bs(2, loss = 2)
        circuit.add_ps(1, 0.7, loss = 0.5)
        circuit.add_ps(3, 0.6, loss = 0.5)
        circuit.add_bs(1, loss = 1.3)
        circuit.add_bs(2, loss = 2)
        circuit.add_ps(1, 0.5, loss = 0.5)
        sampler = QuickSampler(circuit, State([1,0,1,0]), 
                               photon_counting = False, 
                               herald = lambda s: s[0] == 0)
        p = sampler.probability_distribution[State([0,1,1,0])]
        self.assertAlmostEqual(p, 0.386272843449, 8)
        
    def test_circuit_update_with_sampler(self):
        """
        Checks that when a circuit is modified then the sampler recalculates 
        the probability distribution.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1,0,1,0]))
        p1 = sampler.probability_distribution
        circuit.add_bs(0)
        circuit.add_bs(2)
        p2 = sampler.probability_distribution
        self.assertNotEqual(p1, p2)
        
    def test_circuit_parameter_update_with_sampler(self):
        """
        Checks that when the parameters of a circuit are updated then the 
        corresponding probability distribution is modified.
        """
        p = Parameter(0.3)
        circuit = Circuit(4)
        circuit.add_bs(0, reflectivity = p)
        circuit.add_bs(2, reflectivity = p)
        circuit.add_bs(1, reflectivity = p)
        sampler = QuickSampler(circuit, State([1,0,1,0]))
        p1 = sampler.probability_distribution
        p.set(0.7)
        p2 = sampler.probability_distribution
        self.assertNotEqual(p1, p2)
        
    def test_input_update_with_sampler(self):
        """
        Confirms that changing the input state to the sampler alters the 
        produced results.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1,0,1,0]))
        p1 = sampler.probability_distribution
        sampler.input_state = State([0,1,0,1])
        p2 = sampler.probability_distribution
        self.assertNotEqual(p1, p2)
        
    def test_photon_counting_update_with_sampler(self):
        """
        Confirms that changing the photon_counting attribute changes the 
        produced results.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1,0,1,0]), 
                               photon_counting = True)
        p1 = sampler.probability_distribution
        sampler.photon_counting = False
        p2 = sampler.probability_distribution
        self.assertNotEqual(p1, p2)
        
    def test_herald_update_with_sampler(self):
        """
        Confirms that changing the herald function changes the produced 
        results.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1,0,1,0]), 
                               herald = lambda s: s[0] == 1)
        p1 = sampler.probability_distribution
        sampler.herald = lambda s: s[1] == 1
        p2 = sampler.probability_distribution
        self.assertNotEqual(p1, p2)
    
    def test_circuit_assignment(self):
        """
        Confirms that a Circuit cannot be replaced with a non-Circuit through
        the circuit attribute.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1,0,1,0]))
        with self.assertRaises(TypeError):
            sampler.circuit = random_unitary(4)
                
    def test_input_assignmnet(self):
        """
        Checks that the input state of the sampler cannot be assigned to a 
        non-State value and requires the correct number of modes.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1,0,1,0]))
        # Incorrect type
        with self.assertRaises(TypeError):
            sampler.input_state = [1,2,3,4]
        # Incorrect number of modes
        with self.assertRaises(StateError):
            sampler.input_state = State([1,2,3])
            
    def test_herald_assignment(self):
        """
        Confirms that the herald attribute cannot be replaced with a 
        non-function value.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1,0,1,0]))
        with self.assertRaises(TypeError):
            sampler.herald = True
            
    def test_photon_counting_assignment(self):
        """
        Confirms that the photon_counting attribute cannot be replaced with a 
        non-boolean value.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1,0,1,0]))
        with self.assertRaises(TypeError):
            sampler.photon_counting = 1
            
    def test_fermionic_sampling_output(self):
        """
        Checks that fermionic sampling behaves as expected, returning only 
        single photon states.
        """
        set_statistic_type("fermionic")
        circuit = Unitary(random_unitary(4))
        # Create sampler and get results
        sampler = QuickSampler(circuit, State([1,1,1,0]))
        results = sampler.sample_N_outputs(10000)
        # Then check no multi-photon states are present        
        for s in results:
            if max(s) > 1:
                self.fail("Multiple photons present in a single mode.")
        # Reset statistics to bosonic
        set_statistic_type("bosonic")
        
    def test_fermionic_sampling_values(self):
        """Checks that fermionic sampling results remain consistent."""
        set_statistic_type("fermionic")
        circuit = Unitary(random_unitary(4, seed = 99))
        # Create sampler and check calculated probability is correct
        sampler = QuickSampler(circuit, State([1,0,1,0]))
        p = sampler.probability_distribution[State([1,0,1,0])]
        self.assertAlmostEqual(p, 0.1846147449203965, 8)
        # Reset statistics to bosonic
        set_statistic_type("bosonic")
        
    def test_invalid_fermionic_sampling(self):
        """
        Confirm an error is raised when an invalid input is used for fermionic 
        sampling.
        """
        set_statistic_type("fermionic")
        circuit = Unitary(random_unitary(4, seed = 99))
        # Create sampler and calculate probability distribution to raise error
        sampler = QuickSampler(circuit, State([2,0,0,0]))
        with self.assertRaises(ValueError):
            sampler.probability_distribution
        # Reset statistics to bosonic
        set_statistic_type("bosonic")
        
    def tearDown(self) -> None:
        """Reset stats to bosonic after all calculation."""
        set_statistic_type("bosonic")
        return super().tearDown()
    
if __name__ == "__main__":
    
    unittest.main()