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
from lightworks.emulator.results import SamplingResult, SimulationResult 

import pytest
from numpy import array
import matplotlib
import matplotlib.pyplot as plt

class TestSamplingResult:
    """Unit tests for SamplingResult object."""
    
    def setup_class(self) -> None:
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
        assert r[State([0,1,0,1])] == 0.4
    
    def test_items(self):
        """Test return value from items method is correct."""
        r = SamplingResult(self.test_dict, self.test_input)
        assert r.items() == self.test_dict.items()
    
    def test_threshold_mapping(self):
        """Check threshold mapping returns the correct result."""
        r = SamplingResult(self.test_dict, self.test_input)
        r2 = r.apply_threshold_mapping()
        # Round results returned from mapping and compare
        out_dict = {s:round(p,4) for s, p in r2.items()}
        assert out_dict == {State([1,0,0,1]) : 0.3, State([0,1,0,1]) : 0.6,
                            State([0,0,1,0]) : 0.3}
    
    def test_parity_mapping(self):
        """Check parity mapping returns the correct result."""
        r = SamplingResult(self.test_dict, self.test_input)
        r2 = r.apply_parity_mapping()
        # Round results returned from mapping and compare
        out_dict = {s:round(p,4) for s, p in r2.items()}
        assert out_dict == {State([1,0,0,1]): 0.3, State([0,1,0,1]): 0.6, 
                            State([0,0,0,0]): 0.3}
        
    def test_single_input_plot(self):
        """
        Confirm plotting is able to work without errors for single input case.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        r.plot()
        plt.close()
            
class TestSimulationResult:
    """Unit tests for SimulationResult object."""
    
    @pytest.fixture(autouse=True)
    def setup_data(self) -> None:
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
        assert r[State([1,1,0,0])] == {State([1,0,1,0]): 0.3, 
                                       State([0,1,0,1]): 0.2, 
                                       State([0,0,2,0]): 0.1}
        
    def test_result_slicing(self):
        """
        Confirms that result retrieval works correctly for multi input case,
        with slicing used to specify the input and output.
        """
        r = SimulationResult(self.test_array, "probability_amplitude", 
                             inputs = self.test_inputs, 
                             outputs = self.test_outputs)
        assert r[State([1,1,0,0]):State([0,0,2,0])] == 0.1
        # Also check open ended slicing
        assert r[State([1,1,0,0]):] == {State([1,0,1,0]): 0.3, 
                                        State([0,1,0,1]): 0.2, 
                                        State([0,0,2,0]): 0.1}
            
    def test_multi_input_plot(self):
        """
        Confirm plotting is able to work without errors for multi input case.
        """
        # NOTE: There is a non intermittent issue that occurs during testing
        # with the subplots method in mpl. This can be fixed by altering the
        # backend to Agg for these tests. Issue noted here:
        # https://stackoverflow.com/questions/71443540/intermittent-pytest-failures-complaining-about-missing-tcl-files-even-though-the
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        r = SimulationResult(self.test_array, "probability_amplitude", 
                             inputs = self.test_inputs, 
                             outputs = self.test_outputs)
        # Test initial plot
        r.plot()
        plt.close()
        # Test plot with conv_to_probability_option
        r.plot(conv_to_probability=True)
        plt.close()
        # Reset backend after test
        matplotlib.use(original_backend)