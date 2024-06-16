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

from lightworks import State, Circuit, Unitary, random_unitary, Parameter
from lightworks.emulator import Analyzer

import pytest


class TestAnalyzer:
    """
    Unit tests to check results produced by Analyzer object in the emulator.
    """

    def setup_method(self) -> None:
        """Create a non-lossy and a lossy circuit for use."""
        self.circuit = Circuit(4)
        self.lossy_circuit = Circuit(4)
        for i, m in enumerate([0, 2, 1, 2, 0, 1]):
            self.circuit.add_bs(m)
            self.circuit.add_ps(m, phi=i)
            self.circuit.add_bs(m)
            self.circuit.add_ps(m + 1, phi=3 * i)
            # lossy circuit
            self.lossy_circuit.add_bs(m, loss=1 + 0.2 * i)
            self.lossy_circuit.add_ps(m, phi=i)
            self.lossy_circuit.add_bs(m, loss=0.6 + 0.1 * i)
            self.lossy_circuit.add_ps(m + 1, phi=3 * i)

    def test_hom(self):
        """Checks basic hom and confirms probability of |2,0> is 0.5."""
        circuit = Circuit(2)
        circuit.add_bs(0)
        analyzer = Analyzer(circuit)
        results = analyzer.analyze(State([1, 1]))
        p = results[State([2, 0])]
        assert pytest.approx(p) == 0.5

    def test_known_result(self):
        """
        Builds a circuit which produces a known result and checks this is found
        at the output.
        """
        # Build circuit
        circuit = Circuit(4)
        circuit.add_bs(1, reflectivity=0.6)
        circuit.add_mode_swaps({0: 1, 1: 0, 2: 3, 3: 2})
        circuit.add_bs(0, 3, reflectivity=0.3)
        circuit.add_bs(0)
        # And check output counts
        analyzer = Analyzer(circuit)
        results = analyzer.analyze(State([1, 0, 0, 1]))
        assert pytest.approx(abs(results[State([0, 1, 1, 0])])) == 0.5

    def test_analyzer_basic(self):
        """Check analyzer result with basic circuit."""
        analyzer = Analyzer(self.circuit)
        results = analyzer.analyze(State([1, 0, 1, 0]))
        p = results[State([0, 1, 0, 1])]
        assert pytest.approx(p, 1e-8) == 0.6331805740170607

    def test_analyzer_basic_2photons_in_mode(self):
        """Check analyzer result with basic circuit."""
        analyzer = Analyzer(self.circuit)
        results = analyzer.analyze(State([2, 0, 0, 0]))
        p = results[State([0, 1, 0, 1])]
        assert pytest.approx(p, 1e-8) == 0.0022854516590

    def test_analyzer_complex(self):
        """Check analyzer result when using post-selection and heralding."""
        # Add heralding mode
        self.circuit.add_herald(0, 3)
        analyzer = Analyzer(self.circuit)
        # Just heralding
        results = analyzer.analyze(State([1, 0, 1]))
        p = results[State([0, 1, 1])]
        assert pytest.approx(p, 1e-8) == 0.091713377373246
        # Heralding + post-selection
        analyzer.set_post_selection(lambda s: s[0] == 1)
        results = analyzer.analyze(State([1, 0, 1]))
        p = results[State([1, 1, 0])]
        assert pytest.approx(p, 1e-8) == 0.002934140618653
        # Check performance metric
        assert pytest.approx(results.performance, 1e-8) == 0.03181835438235

    def test_analyzer_complex_lossy(self):
        """
        Check analyzer result when using post-selection and heralding with a
        lossy circuit.
        """
        # Add heralding mode
        self.lossy_circuit.add_herald(0, 3)
        analyzer = Analyzer(self.lossy_circuit)
        # Just heralding
        results = analyzer.analyze(State([1, 0, 1]))
        p = results[State([0, 1, 0])]
        assert pytest.approx(p, 1e-8) == 0.062204471804458
        # Heralding + post-selection
        analyzer.set_post_selection(lambda s: s[0] == 0)
        results = analyzer.analyze(State([1, 0, 1]))
        p = results[State([0, 0, 1])]
        assert pytest.approx(p, 1e-8) == 0.0202286624257920
        p = results[State([0, 0, 0])]
        assert pytest.approx(p, 1e-8) == 0.6051457174354371
        # Check performance metric
        assert pytest.approx(results.performance, 1e-8) == 0.6893563871958014

    def test_analyzer_complex_lossy_added_circuit(self):
        """
        Check analyzer result when using post-selection and heralding with a
        lossy circuit which has been added to another circuit.
        """
        # Add heralding mode
        self.lossy_circuit.add_herald(0, 3)
        new_circ = Circuit(
            self.lossy_circuit.n_modes
            - len(self.lossy_circuit.heralds["input"])
        )
        new_circ.add(self.lossy_circuit)
        analyzer = Analyzer(new_circ)
        # Just heralding
        results = analyzer.analyze(State([1, 0, 1]))
        p = results[State([0, 1, 0])]
        assert pytest.approx(p, 1e-8) == 0.062204471804458
        # Heralding + post-selection
        analyzer.set_post_selection(lambda s: s[0] == 0)
        results = analyzer.analyze(State([1, 0, 1]))
        p = results[State([0, 0, 1])]
        assert pytest.approx(p, 1e-8) == 0.0202286624257920
        p = results[State([0, 0, 0])]
        assert pytest.approx(p, 1e-8) == 0.6051457174354371
        # Check performance metric
        assert pytest.approx(results.performance, 1e-8) == 0.6893563871958014

    def test_analyzer_error_rate(self):
        """Check the calculated error rate is correct for a given situation."""
        analyzer = Analyzer(self.circuit)
        expectations = {
            State([1, 0, 1, 0]): State([0, 1, 0, 1]),
            State([0, 1, 0, 1]): State([1, 0, 1, 0]),
        }
        results = analyzer.analyze(
            [State([1, 0, 1, 0]), State([0, 1, 0, 1])], expected=expectations
        )
        assert pytest.approx(results.error_rate, 1e-8) == 0.46523865112110574

    def test_analyzer_circuit_update(self):
        """Check analyzer result before and after a circuit is modified."""
        circuit = Unitary(random_unitary(4))
        # Create analyzer and get results
        analyzer = Analyzer(circuit)
        analyzer.set_post_selection(lambda s: s[0] == 1)
        results = analyzer.analyze(State([1, 0, 1, 0]))
        p = results[State([1, 1, 0, 0])]
        # Update circuit and get results
        circuit.add_bs(0)
        results = analyzer.analyze(State([1, 0, 1, 0]))
        p2 = results[State([1, 1, 0, 0])]
        assert p != p2

    def test_analyzer_circuit_parameter_update(self):
        """
        Check analyzer result before and after a circuit parameters is
        modified.
        """
        param = Parameter(0.3)
        circuit = Circuit(4)
        circuit.add_bs(0, reflectivity=param)
        circuit.add_bs(2, reflectivity=param)
        circuit.add_bs(1, reflectivity=param)
        # Create analyzer and get results
        analyzer = Analyzer(circuit)
        analyzer.set_post_selection(lambda s: s[0] == 1)
        results = analyzer.analyze(State([1, 0, 1, 0]))
        p = results[State([1, 1, 0, 0])]
        # Update parameter and get results
        param.set(0.65)
        results = analyzer.analyze(State([1, 0, 1, 0]))
        p2 = results[State([1, 1, 0, 0])]
        assert p != p2

    def test_circuit_assignment(self):
        """
        Checks that an incorrect value cannot be assigned to the circuit
        attribute.
        """
        circuit = Unitary(random_unitary(4))
        analyzer = Analyzer(circuit)
        with pytest.raises(TypeError):
            analyzer.circuit = random_unitary(5)
