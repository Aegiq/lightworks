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

import matplotlib
import matplotlib.pyplot as plt
import pytest
from numpy import array

from lightworks import ResultCreationError, State
from lightworks.sdk.results import (
    SamplingResult,
    SimulationResult,
)


class TestSamplingResult:
    """Unit tests for SamplingResult object."""

    def setup_class(self) -> None:
        """Create a variety of useful pieces of data for testing."""
        self.test_input = State([1, 1, 0, 0])
        self.test_dict = {
            State([1, 0, 0, 1]): 0.3,
            State([0, 1, 0, 1]): 0.4,
            State([0, 0, 2, 0]): 0.3,
            State([0, 3, 0, 1]): 0.2,
        }

    def test_dict_result_creation(self):
        """
        Checks that a result object can be created with a dictionary
        successfully.
        """
        SamplingResult(self.test_dict, self.test_input)

    def test_invalid_input_type(self):
        """
        Checks that input state is required to be a State object.
        """
        with pytest.raises(ResultCreationError):
            SamplingResult(self.test_dict, self.test_input.s)

    def test_single_input_retrival(self):
        """
        Confirms that result retrieval works correctly for single input case.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        assert r[State([0, 1, 0, 1])] == 0.4

    def test_items(self):
        """Test return value from items method is correct."""
        r = SamplingResult(self.test_dict, self.test_input)
        assert r.items() == self.test_dict.items()

    def test_keys(self):
        """
        Checks that keys attribute returns a list of the output states.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        assert list(r.keys()) == r.outputs

    def test_threshold_mapping(self):
        """Check threshold mapping returns the correct result."""
        r = SamplingResult(self.test_dict, self.test_input)
        r2 = r.apply_threshold_mapping()
        # Round results returned from mapping and compare
        out_dict = {s: round(p, 4) for s, p in r2.items()}
        assert out_dict == {
            State([1, 0, 0, 1]): 0.3,
            State([0, 1, 0, 1]): 0.6,
            State([0, 0, 1, 0]): 0.3,
        }

    def test_parity_mapping(self):
        """Check parity mapping returns the correct result."""
        r = SamplingResult(self.test_dict, self.test_input)
        r2 = r.apply_parity_mapping()
        # Round results returned from mapping and compare
        out_dict = {s: round(p, 4) for s, p in r2.items()}
        assert out_dict == {
            State([1, 0, 0, 1]): 0.3,
            State([0, 1, 0, 1]): 0.6,
            State([0, 0, 0, 0]): 0.3,
        }

    def test_single_input_plot(self):
        """
        Confirm plotting is able to work without errors for single input case.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        r.plot(show=False)
        plt.close()

    def test_extra_attribute_assignment(self):
        """
        Check that miscellaneous additional kwargs can be provided in the
        initial function call and that these are assigned to attributes.
        """
        r = SamplingResult(self.test_dict, self.test_input, test_attr=2.5)
        assert r.test_attr == 2.5

    def test_print_outputs(self):
        """
        Checks no exceptions are raised when the print outputs method is
        called.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        r.print_outputs()

    def test_display_as_dataframe(self):
        """
        Checks that no exceptions are raised when display as Dataframe method
        is called.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        r.display_as_dataframe()

    def test_iterable(self):
        """
        Checks that SamplingResult acts an iterable which will run through all
        outputs included in the result.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        values = []
        for i in r:
            values.append(i)
        assert values == r.outputs

    def test_str(self):
        """
        Tests that string return should be a string representation of the
        results dictionary
        """
        r = SamplingResult(self.test_dict, self.test_input)
        assert str(r) == str(self.test_dict)

    def test_get_items(self):
        """
        Checks that get item returns correct value for output.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        assert r[State([1, 0, 0, 1])] == 0.3

    def test_get_item_invalid_state(self):
        """
        Checks that get item raises a KeyError when output isn't found.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        with pytest.raises(KeyError):
            r[State([1, 0, 2, 1])]

    @pytest.mark.parametrize("value", [[0, 1, 2, 3], [0], ["|1,0,0,1>"]])
    def test_get_item_invalid_type(self, value):
        """
        Checks that get item returns a TypeError when a non-state key is used.
        """
        r = SamplingResult(self.test_dict, self.test_input)
        with pytest.raises(TypeError):
            r[value]


class TestSimulationResult:
    """Unit tests for SimulationResult object."""

    def setup_method(self) -> None:
        """Create a variety of useful pieces of data for testing."""
        # Single input
        self.test_single_inputs = [State([1, 1, 0, 0])]
        self.test_single_outputs = [
            State([1, 0, 1, 0]),
            State([0, 1, 0, 1]),
            State([1, 0, 3, 0]),
        ]
        self.test_single_array = array([[0.3, 0.2, 0.1]])
        # Multiple inputs
        self.test_multi_inputs = [State([1, 1, 0, 0]), State([0, 0, 1, 1])]
        self.test_multi_outputs = [
            State([1, 0, 1, 0]),
            State([0, 1, 0, 1]),
            State([1, 0, 3, 0]),
        ]
        self.test_multi_array = array([[0.3, 0.2, 0.1], [0.2, 0.4, 0.5]])

    def test_single_array_result_creation(self):
        """
        Checks that a result object can be created with an array successfully
        with a single input.
        """
        SimulationResult(
            self.test_single_array,
            "probability_amplitude",
            inputs=self.test_single_inputs,
            outputs=self.test_single_outputs,
        )

    def test_multi_array_result_creation(self):
        """
        Checks that a result object can be created with an array successfully
        with multiple inputs.
        """
        SimulationResult(
            self.test_multi_array,
            "probability_amplitude",
            inputs=self.test_multi_inputs,
            outputs=self.test_multi_outputs,
        )

    def test_single_input_retrival(self):
        """
        Checks that output can be indexed in case of single input.
        """
        r = SimulationResult(
            self.test_single_array,
            "probability_amplitude",
            inputs=self.test_single_inputs,
            outputs=self.test_single_outputs,
        )
        assert r[self.test_single_inputs[0]][State([1, 0, 1, 0])] == 0.3

    def test_multi_input_retrival(self):
        """
        Confirms that result retrieval works correctly for multi input case.
        """
        r = SimulationResult(
            self.test_multi_array,
            "probability_amplitude",
            inputs=self.test_multi_inputs,
            outputs=self.test_multi_outputs,
        )
        assert r[State([1, 1, 0, 0])] == {
            State([1, 0, 1, 0]): 0.3,
            State([0, 1, 0, 1]): 0.2,
            State([1, 0, 3, 0]): 0.1,
        }

    def test_result_indexing(self):
        """
        Confirms that result retrieval works correctly for multi input case,
        with both the input and output used to get a single value.
        """
        r = SimulationResult(
            self.test_multi_array,
            "probability_amplitude",
            inputs=self.test_multi_inputs,
            outputs=self.test_multi_outputs,
        )
        assert r[State([1, 1, 0, 0]), State([1, 0, 3, 0])] == 0.1

    @pytest.mark.parametrize(
        ("conv_to_probability", "rtype"),
        [
            (True, "probability_amplitude"),
            (False, "probability_amplitude"),
            (True, "probability"),
            (False, "probability"),
        ],
    )
    def test_multi_input_plot(self, conv_to_probability, rtype):
        """
        Confirm plotting is able to work without errors for multi input case.
        """
        # NOTE: There is a non intermittent issue that occurs during testing
        # with the subplots method in mpl. This can be fixed by altering the
        # backend to Agg for these tests. Issue noted here:
        # https://stackoverflow.com/questions/71443540/intermittent-pytest-failures-complaining-about-missing-tcl-files-even-though-the
        original_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        r = SimulationResult(
            self.test_multi_array,
            rtype,
            inputs=self.test_multi_inputs,
            outputs=self.test_multi_outputs,
        )
        # Test plot
        r.plot(conv_to_probability=conv_to_probability, show=False)
        plt.close()
        # Reset backend after test
        matplotlib.use(original_backend)

    @pytest.mark.parametrize(
        ("conv_to_probability", "rtype"),
        [
            (True, "probability_amplitude"),
            (False, "probability_amplitude"),
            (True, "probability"),
            (False, "probability"),
        ],
    )
    def test_single_input_plot(self, conv_to_probability, rtype):
        """
        Confirm plotting is able to work without errors for single input case.
        """
        original_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        r = SimulationResult(
            self.test_single_array,
            rtype,
            inputs=self.test_single_inputs,
            outputs=self.test_single_outputs,
        )
        # Test plot
        r.plot(conv_to_probability=conv_to_probability, show=False)
        plt.close()
        # Reset backend after test
        matplotlib.use(original_backend)

    def test_extra_attribute_assignment(self):
        """
        Check that miscellaneous additional kwargs can be provided in the
        initial function call and that these are assigned to attributes.
        """
        r = SimulationResult(
            self.test_multi_array,
            "probability_amplitude",
            inputs=self.test_multi_inputs,
            outputs=self.test_multi_outputs,
            performance=2.5,
        )
        assert r.performance == 2.5

    def test_single_print_outputs(self):
        """
        Confirms print outputs runs without raising an exception.
        """
        r = SimulationResult(
            self.test_single_array,
            "probability_amplitude",
            inputs=self.test_single_inputs,
            outputs=self.test_single_outputs,
        )
        r.print_outputs()

    def test_multi_print_outputs(self):
        """
        Confirms print outputs runs without raising an exception.
        """
        r = SimulationResult(
            self.test_multi_array,
            "probability_amplitude",
            inputs=self.test_multi_inputs,
            outputs=self.test_multi_outputs,
        )
        r.print_outputs()

    @pytest.mark.parametrize("conv_to_probability", [True, False])
    def test_single_display_as_dataframe(self, conv_to_probability):
        """
        Confirms display as dataframe runs without raising an exception.
        """
        r = SimulationResult(
            self.test_single_array,
            "probability_amplitude",
            inputs=self.test_single_inputs,
            outputs=self.test_single_outputs,
        )
        r.display_as_dataframe(conv_to_probability=conv_to_probability)

    @pytest.mark.parametrize("conv_to_probability", [True, False])
    def test_multi_display_as_dataframe(self, conv_to_probability):
        """
        Confirms display as dataframe runs without raising an exception.
        """
        r = SimulationResult(
            self.test_multi_array,
            "probability_amplitude",
            inputs=self.test_multi_inputs,
            outputs=self.test_multi_outputs,
        )
        r.display_as_dataframe(conv_to_probability=conv_to_probability)

    def test_single_input_parity_mapping(self):
        """
        Tests result is correct when the parity mapping is applied in the
        single input case.
        """
        r = SimulationResult(
            self.test_single_array,
            "probability",
            inputs=self.test_single_inputs,
            outputs=self.test_single_outputs,
        )
        new_r = r.apply_parity_mapping()
        assert new_r == {
            State([1, 1, 0, 0]): {
                State([1, 0, 1, 0]): 0.4,
                State([0, 1, 0, 1]): 0.2,
            }
        }

    def test_multi_input_parity_mapping(self):
        """
        Tests result is correct when the parity mapping is applied in the
        multi input case.
        """
        r = SimulationResult(
            self.test_multi_array,
            "probability",
            inputs=self.test_multi_inputs,
            outputs=self.test_multi_outputs,
        )
        new_r = r.apply_parity_mapping()
        assert new_r == {
            State([1, 1, 0, 0]): {
                State([1, 0, 1, 0]): 0.4,
                State([0, 1, 0, 1]): 0.2,
            },
            State([0, 0, 1, 1]): {
                State([1, 0, 1, 0]): 0.7,
                State([0, 1, 0, 1]): 0.4,
            },
        }

    def test_single_input_threshold_mapping(self):
        """
        Tests result is correct when the threshold mapping is applied in the
        single input case.
        """
        r = SimulationResult(
            self.test_single_array,
            "probability",
            inputs=self.test_single_inputs,
            outputs=self.test_single_outputs,
        )
        new_r = r.apply_threshold_mapping()
        assert new_r == {
            State([1, 1, 0, 0]): {
                State([1, 0, 1, 0]): 0.4,
                State([0, 1, 0, 1]): 0.2,
            }
        }

    def test_multi_input_threshold_mapping(self):
        """
        Tests result is correct when the threshold mapping is applied in the
        single multi case.
        """
        r = SimulationResult(
            self.test_multi_array,
            "probability",
            inputs=self.test_multi_inputs,
            outputs=self.test_multi_outputs,
        )
        new_r = r.apply_threshold_mapping()
        assert new_r == {
            State([1, 1, 0, 0]): {
                State([1, 0, 1, 0]): 0.4,
                State([0, 1, 0, 1]): 0.2,
            },
            State([0, 0, 1, 1]): {
                State([1, 0, 1, 0]): 0.7,
                State([0, 1, 0, 1]): 0.4,
            },
        }
