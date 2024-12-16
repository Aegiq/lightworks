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

# ruff: noqa: E722

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lightworks import (
    Display,
    DisplayError,
    Parameter,
    PhotonicCircuit,
    Unitary,
    db_loss_to_decimal,
    random_unitary,
)


class TestDisplay:
    """Unit testing for display functionality of circuit."""

    def setup_method(self) -> None:
        """
        Create a circuit for testing with, this should utilise all components
        to ensure thorough testing.
        """
        self.circuit = PhotonicCircuit(4)
        for i, m in enumerate([0, 2, 1, 2, 0, 1]):
            self.circuit.bs(m)
            self.circuit.ps(m, phi=Parameter(i, label=f"p{i}"))
            self.circuit.bs(m, loss=db_loss_to_decimal(2))
            self.circuit.ps(m + 1, phi=3 * i)
            self.circuit.loss(m, loss=db_loss_to_decimal(1))
            self.circuit.loss(m, loss=Parameter(1, label="test"))
        self.circuit.bs(0, 3)
        self.circuit.bs(3, 0)
        # Check a number of different phase values can be set
        for val in [
            np.pi / 2,
            np.pi,
            0,
            1.6,
            5 * np.pi / 2,
            np.pi / 4,
            7 * np.pi / 4,
            -7 * np.pi / 4,
        ]:
            self.circuit.ps(2, val)
        self.circuit.barrier([1, 2, 3])
        self.circuit.mode_swaps({})
        self.circuit.mode_swaps({0: 2, 2: 1, 1: 0})
        self.circuit.herald(1, 0, 0)
        self.circuit.add(Unitary(random_unitary(3, seed=1)), 1)
        self.circuit.add(Unitary(random_unitary(3, seed=1)), 0, group=True)
        self.circuit.barrier()
        circuit2 = PhotonicCircuit(2)
        circuit2.bs(0)
        circuit2.ps(1, 2)
        circuit2.herald(2, 1, 1)
        self.circuit.add(circuit2, 1)
        circuit2.add(circuit2, group=True)
        self.circuit.add(circuit2, 1, group=True)

    def test_circuit_display_method(self):
        """
        Checks that the display method works without any errors arising.
        """
        try:
            self.circuit.display(display_loss=True)
        except:
            pytest.fail("Exception occurred during display operation.")

    def test_circuit_display_method_no_loss(self):
        """
        Checks that the display method works without any errors arising when
        display loss is False.
        """
        try:
            self.circuit.display(display_loss=False)
        except:
            pytest.fail("Exception occurred during display operation.")

    def test_circuit_display_show_parameter_values(self):
        """
        Checks that the display method works without any errors arising when
        the show parameter values option is used.
        """
        try:
            self.circuit.display(display_loss=True, show_parameter_values=True)
        except:
            pytest.fail("Exception occurred during display operation.")

    def test_circuit_display_mode_labels(self):
        """
        Checks that the display method works without any errors arising when
        mode labels are specified.
        """
        try:
            self.circuit.display(
                display_loss=True, mode_labels=["a", "b", "c", "d"]
            )
        except:
            pytest.fail("Exception occurred during display operation.")

    def test_circuit_display_function(self):
        """
        Checks that a circuit passed to the display function is able to be
        processed without any exceptions arising.
        """
        try:
            Display(self.circuit, display_loss=True)
        except:
            pytest.fail("Exception occurred during display operation.")

    def test_circuit_display_function_mpl(self):
        """
        Checks that a circuit passed to the display function is able to be
        processed without any exceptions arising for the matplotlib method.
        """
        # NOTE: There is a non intermittent issue that occurs during testing
        # with the subplots method in mpl. This can be fixed by altering the
        # backend to Agg for these tests. Issue noted here:
        # https://stackoverflow.com/questions/71443540/intermittent-pytest-failures-complaining-about-missing-tcl-files-even-though-the
        original_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        try:
            Display(self.circuit, display_loss=True, display_type="mpl")
            plt.close()
        except:
            pytest.fail("Exception occurred during display operation.")
        # Reset backend after test
        matplotlib.use(original_backend)

    def test_display_type_error(self):
        """
        Confirms an error is raised when an invalid display type is passed to
        the display function.
        """
        with pytest.raises(DisplayError):
            self.circuit.display(display_type="not_valid")
        with pytest.raises(DisplayError):
            Display(self.circuit, display_type="not_valid")

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.parametrize("display_type", ["svg", "mpl"])
    def test_incorrect_mode_labels(self, display_type):
        """
        Checks an error is raised in the dimension of the mode labels setting is
        incorrect.
        """
        labels = ["1"] * 20
        with pytest.raises(DisplayError):
            self.circuit.display(display_type=display_type, mode_labels=labels)
