from lightworks import State, Circuit, Unitary, random_unitary, Parameter
from lightworks.emulator import Analyzer, set_statistic_type
import unittest

class AnalyzerTest(unittest.TestCase):
    """
    Unit tests to check results produced by Analyzer object in the emulator.
    """
    
    def setUp(self) -> None:
        """Create a non-lossy and a lossy circuit for use."""
        self.circuit = Circuit(4)
        self.lossy_circuit = Circuit(4)
        for i, m in enumerate([0,2,1,2,0,1]):
            self.circuit.add_bs(m)
            self.circuit.add_ps(m, phi = i)
            self.circuit.add_bs(m)
            self.circuit.add_ps(m+1, phi = 3*i)
            # lossy circuit
            self.lossy_circuit.add_bs(m, loss = 1 + 0.2*i)
            self.lossy_circuit.add_ps(m, phi = i)
            self.lossy_circuit.add_bs(m, loss = 0.6 + 0.1*i)
            self.lossy_circuit.add_ps(m+1, phi = 3*i)
            
    def test_hom(self):
        """Checks basic hom and confirms probability of |2,0> is 0.5."""
        circuit = Circuit(2)
        circuit.add_bs(0)
        analyzer = Analyzer(circuit)
        results = analyzer.analyze(State([1,1]))
        p = results[State([2,0])]
        self.assertAlmostEqual(p, 0.5, 6)
        
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
        # And check output counts
        analyzer = Analyzer(circuit)
        results = analyzer.analyze(State([1,0,0,1]))
        self.assertAlmostEqual(abs(results[State([0,1,1,0])]), 0.5, 8)
    
    def test_analyzer_basic(self):
        """Check analyzer result with basic circuit."""
        analyzer = Analyzer(self.circuit)
        results = analyzer.analyze(State([1,0,1,0]))
        p = results[State([0,1,0,1])]
        self.assertAlmostEqual(p, 0.6331805740170607, 8)
        
    def test_analyzer_basic_2photons_in_mode(self):
        """Check analyzer result with basic circuit."""
        analyzer = Analyzer(self.circuit)
        results = analyzer.analyze(State([2,0,0,0]))
        p = results[State([0,1,0,1])]
        self.assertAlmostEqual(p, 0.0022854516590, 8)
        
    def test_analyzer_complex(self):
        """Check analyzer result when using post-selection and heralding."""
        analyzer = Analyzer(self.circuit)
        # Just heralding
        analyzer.set_herald(3, 0)
        results = analyzer.analyze(State([1,0,1]))
        p = results[State([0,1,1])]
        self.assertAlmostEqual(p, 0.091713377373246, 8)
        # Heralding + post-selection
        analyzer.set_post_selection(lambda s: s[0] == 1)
        results = analyzer.analyze(State([1,0,1]))
        p = results[State([1,1,0])]
        self.assertAlmostEqual(p, 0.002934140618653, 8)
        # Check performance metric
        self.assertAlmostEqual(results.performance, 0.03181835438235, 8)
        
    def test_analyzer_complex_lossy(self):
        """
        Check analyzer result when using post-selection and heralding with a 
        lossy circuit.
        """
        analyzer = Analyzer(self.lossy_circuit)
        # Just heralding
        analyzer.set_herald(3, 0)
        results = analyzer.analyze(State([1,0,1]))
        p = results[State([0,1,0])]
        self.assertAlmostEqual(p, 0.062204471804458, 8)
        # Heralding + post-selection
        analyzer.set_post_selection(lambda s: s[0] == 0)
        results = analyzer.analyze(State([1,0,1]))
        p = results[State([0,0,1])]
        self.assertAlmostEqual(p, 0.0202286624257920, 8)
        p = results[State([0,0,0])]
        self.assertAlmostEqual(p, 0.6051457174354371, 8)
        # Check performance metric
        self.assertAlmostEqual(results.performance, 0.6893563871958014, 8)
        
    def test_analyzer_error_rate(self):
        """Check the calculated error rate is correct for a given situation."""
        analyzer = Analyzer(self.circuit)
        expectations = {State([1,0,1,0]) : State([0,1,0,1]),
                        State([0,1,0,1]) : State([1,0,1,0])}
        results = analyzer.analyze([State([1,0,1,0]), State([0,1,0,1])],
                                   expected = expectations)
        self.assertAlmostEqual(results.error_rate, 0.46523865112110574, 8)
        
    def test_analyzer_circuit_update(self):
        """Check analyzer result before and after a circuit is modified."""
        circuit = Unitary(random_unitary(4))
         # Create analyzer and get results
        analyzer = Analyzer(circuit)
        analyzer.set_post_selection(lambda s: s[0] == 1)
        results = analyzer.analyze(State([1,0,1,0]))
        p = results[State([1,1,0,0])]
        # Update circuit and get results
        circuit.add_bs(0)
        results = analyzer.analyze(State([1,0,1,0]))
        p2 = results[State([1,1,0,0])]
        self.assertNotEqual(p, p2)
        
    def test_analyzer_circuit_parameter_update(self):
        """
        Check analyzer result before and after a circuit parameters is 
        modified.
        """
        param = Parameter(0.3)
        circuit = Circuit(4)
        circuit.add_bs(0, reflectivity = param)
        circuit.add_bs(2, reflectivity = param)
        circuit.add_bs(1, reflectivity = param)
        # Create analyzer and get results
        analyzer = Analyzer(circuit)
        analyzer.set_post_selection(lambda s: s[0] == 1)
        results = analyzer.analyze(State([1,0,1,0]))
        p = results[State([1,1,0,0])]
        # Update parameter and get results
        param.set(0.65)
        results = analyzer.analyze(State([1,0,1,0]))
        p2 = results[State([1,1,0,0])]
        self.assertNotEqual(p, p2)
                
    def test_circuit_assignment(self):
        """
        Checks that an incorrect value cannot be assigned to the circuit 
        attribute.
        """
        circuit = Unitary(random_unitary(4))
        analyzer = Analyzer(circuit)
        with self.assertRaises(TypeError):
            analyzer.circuit = random_unitary(5)
            
    def test_analyzer_fermionic_outputs(self):
        """
        Confirm that the analyzer with fermionic statistics produces only 
        single photon outputs.
        """ 
        set_statistic_type("fermionic")
        # Build analyzer
        analyzer = Analyzer(self.lossy_circuit)
        analyzer.set_herald(3, 0)
        results = analyzer.analyze(State([1,0,1]))
        # Check results
        for s in results[:]:
            if max(s) > 1:
                self.fail("Multi-photon state found in a mode.")
        set_statistic_type("bosonic")
        
    def test_analyzer_fermionic_value(self):
        """
        Confirm that the analyzer with fermionic statistics produces known 
        output values.
        """ 
        set_statistic_type("fermionic")
        # Build analyzer
        analyzer = Analyzer(self.lossy_circuit)
        analyzer.set_herald(3, 0)
        analyzer.set_post_selection(lambda s: s[0] == 0)
        results = analyzer.analyze(State([1,0,1]))
        # Check results
        p = results[State([0,1,0])]
        self.assertAlmostEqual(p, 0.06216965933376453, 8)
        # Also check performance metric
        self.assertAlmostEqual(results.performance, 0.6892747403578104, 8)
        set_statistic_type("bosonic")
        
    def test_analyzer_fermionic_invalid_input(self):
        """
        Check an error is raised when a state with multi-photons on the same
        mode is attempted to be used with the analyzer, and also when a 
        multi-photon state is used for heralding.
        """
        set_statistic_type("fermionic")
        # Build analyzer
        analyzer = Analyzer(self.lossy_circuit)
        # Multi-photon input
        with self.assertRaises(ValueError):
            results = analyzer.analyze(State([2,0,1]))
        # Multi-photon herald
        with self.assertRaises(ValueError):
            analyzer.set_herald(3, 2)
        set_statistic_type("bosonic")
            
    def tearDown(self) -> None:
        """Reset stats to bosonic after all calculation."""
        set_statistic_type("bosonic")
        return super().tearDown()

if __name__ == "__main__":
    unittest.main()