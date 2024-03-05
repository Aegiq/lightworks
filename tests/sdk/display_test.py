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

from lightworks import Display, Circuit, Unitary, random_unitary, Parameter
import unittest
import matplotlib.pyplot as plt

class DisplayTest(unittest.TestCase):
    """Unit testing for display functionality of circuit."""
    
    def setUp(self) -> None:
        """
        Create a circuit for testing with, this should utilise all components 
        to ensure thorough testing.
        """
        self.circuit = Circuit(4)
        for i, m in enumerate([0,2,1,2,0,1]):
            self.circuit.add_bs(m)
            self.circuit.add_ps(m, phi = Parameter(i, label = f"p{i}"))
            self.circuit.add_bs(m, loss = 2)
            self.circuit.add_ps(m+1, phi = 3*i)
            self.circuit.add_loss(m, loss = 1)
        self.circuit.add_mode_swaps({0:2,2:1,1:0})
        self.circuit.add_mode_swaps({0:2,2:1,1:0}, decompose_into_bs = True,
                                    element_loss = 0.5)
        self.circuit.add(Unitary(random_unitary(3, seed=1)), 1)
        self.circuit.add(Unitary(random_unitary(3, seed=1)), 0, group = True)
        circuit2 = Circuit(2)
        circuit2.add_bs(0)
        circuit2.add_ps(1, 2)
        self.circuit.add(circuit2, 1)
        circuit2.add(circuit2, group=True)
        self.circuit.add(circuit2, 1, group = True)
    
    def test_circuit_display_method(self):
        """Checks that the display method works without any errors arising."""
        try:
            self.circuit.display(display_loss = True)
        except:
            self.fail("Exception occurred during display operation.")
            
    def test_circuit_display_show_parameter_values(self):
        """
        Checks that the display method works without any errors arising when 
        the show parameter values option is used.
        """
        try:
            self.circuit.display(display_loss = True, 
                                 show_parameter_values = True)
        except:
            self.fail("Exception occurred during display operation.")
            
    def test_circuit_display_mode_labels(self):
        """
        Checks that the display method works without any errors arising when 
        mode labels are specified.
        """
        try:
            self.circuit.display(display_loss = True, 
                                 mode_labels = ["a", "b", "c", "d"])
        except:
            self.fail("Exception occurred during display operation.")
            
    def test_circuit_display_function(self):
        """
        Checks that a circuit passed to the display function is able to be
        processed without any exceptions arising.
        """
        try:
            Display(self.circuit, display_loss = True)
        except:
            self.fail("Exception occurred during display operation.")
            
    def test_circuit_display_function_mpl(self):
        """
        Checks that a circuit passed to the display function is able to be
        processed without any exceptions arising for the matplotlib method.
        """
        try:
            Display(self.circuit, display_loss = True, display_type = "mpl")
            plt.close()
        except:
            self.fail("Exception occurred during display operation.")
            
    def test_display_type_error(self):
        """
        Confirms an error is raised when an invalid display type is passed to 
        the display function.
        """
        with self.assertRaises(ValueError):
            self.circuit.display(display_type = "not_valid")
        with self.assertRaises(ValueError):
            Display(self.circuit, display_type = "not_valid")

if __name__ == "__main__":
    unittest.main()