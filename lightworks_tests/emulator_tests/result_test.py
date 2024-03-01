from lightworks import State
from lightworks.emulator.results import SamplingResult, SimulationResult 
import unittest
from numpy import array
import matplotlib.pyplot as plt

class SamplingResultTest(unittest.TestCase):
    """Unit tests for SamplingResult object."""
    
    def setUp(self) -> None:
        """Create a variety of useful pieces of data for testing."""
        self.test_input = State([1,1,0,0])
        self.test_dict = {State([1,0,0,1]) : 0.3,
                          State([0,1,0,1]) : 0.4,
                          State([0,0,2,0]) : 0.3,
                          State([0,3,0,1]) : 0.2}

    def test_dict_result_creation(self):
        """
        Checks that a result object can be created with a dictionary 
        successfully.
        """
        SamplingResult(self.test_dict, self.test_input)
    
    def test_single_input_retrival(self):
        """
        Confirms that result retrieval works correctly for single input case.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        self.assertEqual(r[State([0,1,0,1])], 0.4)
    
    def test_items(self):
        """Test return value from items method is correct."""
        r = SamplingResult(self.test_dict, self.test_input)
        self.assertEqual(r.items(), self.test_dict.items())
    
    def test_threshold_mapping(self):
        """Check threshold mapping returns the correct result."""
        r = SamplingResult(self.test_dict, self.test_input)
        r2 = r.apply_threshold_mapping()
        # Round results returned from mapping and compare
        out_dict = {s:round(p,4) for s, p in r2.items()}
        self.assertEqual(out_dict, {State([1,0,0,1]) : 0.3,
                                    State([0,1,0,1]) : 0.6,
                                    State([0,0,1,0]) : 0.3})
    
    def test_parity_mapping(self):
        """Check parity mapping returns the correct result."""
        r = SamplingResult(self.test_dict, self.test_input)
        r2 = r.apply_parity_mapping()
        # Round results returned from mapping and compare
        out_dict = {s:round(p,4) for s, p in r2.items()}
        self.assertEqual(out_dict, {State([1,0,0,1]): 0.3, 
                                    State([0,1,0,1]): 0.6, 
                                    State([0,0,0,0]): 0.3})
        
    def test_single_input_plot(self):
        """
        Confirm plotting is able to work without errors for single input case.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        # Test initial plot
        try:
            r.plot()
            plt.close()
        except:
            self.fail("Exception occurred during plot operation.")
            
class SimulationResultTest(unittest.TestCase):
    """Unit tests for SimulationResult object."""
    
    def setUp(self) -> None:
        """Create a variety of useful pieces of data for testing."""
        self.test_inputs = [State([1,1,0,0]), State([0,0,1,1])]
        self.test_outputs = [State([1,0,1,0]), State([0,1,0,1]), 
                             State([0,0,2,0])]
        self.test_array = array([[0.3, 0.2, 0.1], [0.2, 0.4, 0.5]])
    
    def test_array_result_creation(self):
        """
        Checks that a result object can be created with an array successfully.
        """
        SimulationResult(self.test_array, "probability_amplitude",
                         inputs = self.test_inputs, 
                         outputs = self.test_outputs)
            
    def test_multi_input_retrival(self):
        """
        Confirms that result retrieval works correctly for multi input case.
        """
        r = SimulationResult(self.test_array, "probability_amplitude", 
                             inputs = self.test_inputs, 
                             outputs = self.test_outputs)
        self.assertEqual(r[State([1,1,0,0])], {State([1,0,1,0]): 0.3, 
                                               State([0,1,0,1]): 0.2, 
                                               State([0,0,2,0]): 0.1})
        
    def test_result_slicing(self):
        """
        Confirms that result retrieval works correctly for multi input case,
        with slicing used to specify the input and output.
        """
        r = SimulationResult(self.test_array, "probability_amplitude", 
                             inputs = self.test_inputs, 
                             outputs = self.test_outputs)
        self.assertEqual(r[State([1,1,0,0]):State([0,0,2,0])], 0.1)
        # Also check open ended slicing
        self.assertEqual(r[State([1,1,0,0]):], {State([1,0,1,0]): 0.3, 
                                                State([0,1,0,1]): 0.2, 
                                                State([0,0,2,0]): 0.1})
            
    def test_multi_input_plot(self):
        """
        Confirm plotting is able to work without errors for multi input case.
        """
        r = SimulationResult(self.test_array, "probability_amplitude", 
                             inputs = self.test_inputs, 
                             outputs = self.test_outputs)
        # Test initial plot
        try:
            r.plot()
            plt.close()
        except:
            self.fail("Exception occurred during plot operation.")
        # Test plot with conv_to_probability_option
        try:
            r.plot(conv_to_probability=True)
            plt.close()
        except:
            self.fail("Exception occurred during second plot operation.")
              
        
if __name__ == "__main__":
    unittest.main()