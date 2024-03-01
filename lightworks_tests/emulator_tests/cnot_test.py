from lightworks import State, Circuit
from lightworks.emulator import Sampler, Source, Detector 
import unittest
from numpy import arccos, pi

class CNOTTest(unittest.TestCase):
    """
    Samples from a non-heralded CNOT circuit with loss, imperfect source and 
    detector to check that the correct results is produced.  
    """
    
    def test_cnot(self):
        """
        Checks the correct output is produced from the CNOT gate. Note, very
        occasionally this test may fail due to the probabilistic nature of the
        sampler. It is only a concern if this happens consistently.
        """
        # Create CNOT circuit
        r = 1/3
        loss = 0.1
        theta = arccos(r)
        cnot_circuit = Circuit(6)
        to_add = [(3, pi/2, 0), (0, theta, 0), (2, theta, pi), (4, theta, 0), 
                (3, pi/2, 0)]
        for m, t, p in to_add:
            cnot_circuit.add_bs(m, loss = loss, reflectivity = 0.5)
            cnot_circuit.add_ps(m+1, t)
            cnot_circuit.add_bs(m, loss = loss, reflectivity = 0.5)
            cnot_circuit.add_ps(m+1, p)
            if m in [3,4,3]:
                cnot_circuit.add_separation()
        # Define imperfect source and detector
        source = Source(purity = 0.99, brightness = 0.4, 
                        indistinguishability = 0.94)
        detector = Detector(efficiency = 0.9, p_dark = 1e-5, 
                            photon_counting = False)
        # Then define sampler with the input state |10>
        sampler = Sampler(cnot_circuit, State([0,0,1,1,0,0]), source = source, 
                          detector = detector)
        # Can then define the heralding required to get the correct output and
        # generate a set of samples
        herald = lambda s: (s[0] == 0 and s[5] == 0 and s[1] + s[2] == 1 and 
                            s[3] + s[4] == 1)
        results = sampler.sample_N_states(20000, herald = herald)
        # We expect the state |11> (|0,0,1,0,1,0> in mode language) with 
        # reasonable fidelity, so we will assert this is measured for > 80% of
        # the total samples which met the herald condition
        eff = results[State([0,0,1,0,1,0])] / sum(results.values())
        self.assertTrue(eff > 0.8) 

if __name__ == "__main__":
    unittest.main()