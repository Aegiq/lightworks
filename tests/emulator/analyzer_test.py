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

import numpy as np
import pytest

from lightworks import (
    Analyzer,
    Parameter,
    PhotonicCircuit,
    PhotonNumberError,
    PostSelection,
    State,
    Unitary,
    convert,
    random_unitary,
)
from lightworks.emulator import Backend

BACKEND = Backend("permanent")


class TestAnalyzer:
    """
    Unit tests to check results produced by Analyzer object in the emulator.
    """

    def setup_method(self) -> None:
        """Create a non-lossy and a lossy circuit for use."""
        self.circuit = PhotonicCircuit(4)
        self.lossy_circuit = PhotonicCircuit(4)
        for i, m in enumerate([0, 2, 1, 2, 0, 1]):
            self.circuit.bs(m)
            self.circuit.ps(m, phi=i)
            self.circuit.bs(m)
            self.circuit.ps(m + 1, phi=3 * i)
            # lossy circuit
            self.lossy_circuit.bs(
                m, loss=convert.db_loss_to_decimal(1 + 0.2 * i)
            )
            self.lossy_circuit.ps(m, phi=i)
            self.lossy_circuit.bs(
                m, loss=convert.db_loss_to_decimal(0.6 + 0.1 * i)
            )
            self.lossy_circuit.ps(m + 1, phi=3 * i)

    def test_hom(self):
        """Checks basic hom and confirms probability of |2,0> is 0.5."""
        circuit = PhotonicCircuit(2)
        circuit.bs(0)
        analyzer = Analyzer(circuit, State([1, 1]))
        results = BACKEND.run(analyzer)[State([1, 1])]
        p = results[State([2, 0])]
        assert pytest.approx(p) == 0.5

    def test_known_result(self):
        """
        Builds a circuit which produces a known result and checks this is found
        at the output.
        """
        # Build circuit
        circuit = PhotonicCircuit(4)
        circuit.bs(1, reflectivity=0.6)
        circuit.mode_swaps({0: 1, 1: 0, 2: 3, 3: 2})
        circuit.bs(0, 3, reflectivity=0.3)
        circuit.bs(0)
        # And check output counts
        analyzer = Analyzer(circuit, State([1, 0, 0, 1]))
        results = BACKEND.run(analyzer)[State([1, 0, 0, 1])]
        assert pytest.approx(abs(results[State([0, 1, 1, 0])])) == 0.5

    def test_analyzer_basic(self):
        """Check analyzer result with basic circuit."""
        analyzer = Analyzer(self.circuit, State([1, 0, 1, 0]))
        results = BACKEND.run(analyzer)[State([1, 0, 1, 0])]
        p = results[State([0, 1, 0, 1])]
        assert pytest.approx(p, 1e-8) == 0.6331805740170607

    def test_analyzer_basic_2photons_in_mode(self):
        """Check analyzer result with basic circuit."""
        analyzer = Analyzer(self.circuit, State([2, 0, 0, 0]))
        results = BACKEND.run(analyzer)[State([2, 0, 0, 0])]
        p = results[State([0, 1, 0, 1])]
        assert pytest.approx(p, 1e-8) == 0.0022854516590

    def test_analyzer_complex(self):
        """Check analyzer result when using post-selection and heralding."""
        # Add heralding mode
        self.circuit.herald(3, 0)
        analyzer = Analyzer(self.circuit, State([1, 0, 1]))
        # Just heralding
        results = BACKEND.run(analyzer)
        p = results[State([1, 0, 1]), State([0, 1, 1])]
        assert pytest.approx(p, 1e-8) == 0.091713377373246
        # Heralding + post-selection
        analyzer.post_selection = lambda s: s[0] == 1
        results = BACKEND.run(analyzer)
        p = results[State([1, 0, 1]), State([1, 1, 0])]
        assert pytest.approx(p, 1e-8) == 0.002934140618653
        # Check performance metric
        assert pytest.approx(results.performance, 1e-8) == 0.03181835438235

    def test_analyzer_complex_lossy(self):
        """
        Check analyzer result when using post-selection and heralding with a
        lossy circuit.
        """
        # Add heralding mode
        self.lossy_circuit.herald(3, 0)
        analyzer = Analyzer(self.lossy_circuit, State([1, 0, 1]))
        # Just heralding
        results = BACKEND.run(analyzer)
        p = results[State([1, 0, 1]), State([0, 1, 0])]
        assert pytest.approx(p, 1e-8) == 0.062204471804458
        # Heralding + post-selection
        analyzer.post_selection = lambda s: s[0] == 0
        results = BACKEND.run(analyzer)
        p = results[State([1, 0, 1]), State([0, 0, 1])]
        assert pytest.approx(p, 1e-8) == 0.0202286624257920
        p = results[State([1, 0, 1]), State([0, 0, 0])]
        assert pytest.approx(p, 1e-8) == 0.6051457174354371
        # Check performance metric
        assert pytest.approx(results.performance, 1e-8) == 0.6893563871958014

    def test_analyzer_complex_lossy_added_circuit(self):
        """
        Check analyzer result when using post-selection and heralding with a
        lossy circuit which has been added to another circuit.
        """
        # Add heralding mode
        self.lossy_circuit.herald(3, 0)
        new_circ = PhotonicCircuit(
            self.lossy_circuit.n_modes - len(self.lossy_circuit.heralds.input)
        )
        new_circ.add(self.lossy_circuit)
        analyzer = Analyzer(new_circ, State([1, 0, 1]))
        # Just heralding
        results = BACKEND.run(analyzer)
        p = results[State([1, 0, 1]), State([0, 1, 0])]
        assert pytest.approx(p, 1e-8) == 0.062204471804458
        # Heralding + post-selection
        analyzer.post_selection = lambda s: s[0] == 0
        results = BACKEND.run(analyzer)
        p = results[State([1, 0, 1]), State([0, 0, 1])]
        assert pytest.approx(p, 1e-8) == 0.0202286624257920
        p = results[State([1, 0, 1]), State([0, 0, 0])]
        assert pytest.approx(p, 1e-8) == 0.6051457174354371
        # Check performance metric
        assert pytest.approx(results.performance, 1e-8) == 0.6893563871958014

    def test_analyzer_error_rate(self):
        """Check the calculated error rate is correct for a given situation."""
        expectations = {
            State([1, 0, 1, 0]): State([0, 1, 0, 1]),
            State([0, 1, 0, 1]): State([1, 0, 1, 0]),
        }
        analyzer = Analyzer(
            self.circuit,
            [State([1, 0, 1, 0]), State([0, 1, 0, 1])],
            expected=expectations,
        )
        results = BACKEND.run(analyzer)
        assert pytest.approx(results.error_rate, 1e-8) == 0.46523865112110574

    def test_analyzer_circuit_update(self):
        """Check analyzer result before and after a circuit is modified."""
        circuit = Unitary(random_unitary(4))
        # Create analyzer and get results
        analyzer = Analyzer(circuit, State([1, 0, 1, 0]))
        analyzer.post_selection = lambda s: s[0] == 1
        results = BACKEND.run(analyzer)
        p = results[State([1, 0, 1, 0]), State([1, 1, 0, 0])]
        # Update circuit and get results
        circuit.bs(0)
        results = BACKEND.run(analyzer)
        p2 = results[State([1, 0, 1, 0]), State([1, 1, 0, 0])]
        assert p != p2

    def test_analyzer_circuit_parameter_update(self):
        """
        Check analyzer result before and after a circuit parameters is
        modified.
        """
        param = Parameter(0.3)
        circuit = PhotonicCircuit(4)
        circuit.bs(0, reflectivity=param)
        circuit.bs(2, reflectivity=param)
        circuit.bs(1, reflectivity=param)
        # Create analyzer and get results
        analyzer = Analyzer(circuit, State([1, 0, 1, 0]))
        post_select = PostSelection()
        post_select.add(0, 1)
        analyzer.post_selection = post_select
        results = BACKEND.run(analyzer)
        p = results[State([1, 0, 1, 0]), State([1, 1, 0, 0])]
        # Update parameter and get results
        param.set(0.65)
        results = BACKEND.run(analyzer)
        p2 = results[State([1, 0, 1, 0]), State([1, 1, 0, 0])]
        assert p != p2

    def test_circuit_assignment(self):
        """
        Checks that an incorrect value cannot be assigned to the circuit
        attribute.
        """
        circuit = Unitary(random_unitary(4))
        analyzer = Analyzer(circuit, State([1, 0, 1, 0]))
        with pytest.raises(TypeError):
            analyzer.circuit = random_unitary(5)

    def test_ns_gate(self):
        """
        Checks case of NS gate which was previously found to cause a bug.
        """
        u_ns = np.array(
            [
                [1 - 2**0.5, 2**-0.25, (3 / (2**0.5) - 2) ** 0.5],
                [2**-0.25, 0.5, 0.5 - 2**-0.5],
                [(3 / (2**0.5) - 2) ** 0.5, 0.5 - 2**-0.5, 2**0.5 - 0.5],
            ]
        )
        ns_gate = Unitary(u_ns)
        ns_gate.herald(1, 1)
        ns_gate.herald(2, 0)
        in_state = State([1])
        # Run simulation
        analyzer = Analyzer(ns_gate, in_state)
        results = BACKEND.run(analyzer)
        # Validate success probability is 0.25
        assert results[in_state, in_state] == pytest.approx(0.25)

    @pytest.mark.parametrize("n_output", [0, 1, 2])
    def test_herald_equivalance(self, n_output):
        """
        Checks that heralded and non-heralded circuits produce the same results.
        """
        # Create heralded and versions of same circuit
        unitary = random_unitary(6)
        circuit = Unitary(unitary)
        circuit_herald = Unitary(unitary)
        circuit_herald.herald((2, 2), 0)
        circuit_herald.herald((1, 3), (1, n_output))
        # Simulate both with equivalent inputs
        sim = Analyzer(circuit, State([0, 1, 0, 1, 1, 0]))
        results = BACKEND.run(sim)
        sim_h = Analyzer(circuit_herald, State([0, 1, 1, 0]))
        results_h = BACKEND.run(sim_h)
        # Then check equivalence of results for all outputs
        for output in results_h.outputs:
            full_state = output[0:2] + State([0, n_output]) + output[2:]
            assert (
                pytest.approx(results_h[State([0, 1, 1, 0]), output])
                == results[State([0, 1, 0, 1, 1, 0]), full_state]
            )

    @pytest.mark.parametrize("n_output", [0, 1, 2])
    def test_herald_equivalance_lossy(self, n_output):
        """
        Checks that heralded and non-heralded circuits produce the same results
        when a lossy circuit is used.
        """
        # Create heralded and versions of same circuit
        unitary = random_unitary(6)
        circuit = Unitary(unitary)
        for i in range(6):
            circuit.loss(i, (i + 1) / 10)
        circuit_herald = Unitary(unitary)
        for i in range(6):
            circuit_herald.loss(i, (i + 1) / 10)
        circuit_herald.herald((2, 2), 0)
        circuit_herald.herald((1, 3), (1, n_output))
        # Simulate both with equivalent inputs
        sim = Analyzer(circuit, State([0, 1, 0, 1, 1, 0]))
        results = BACKEND.run(sim)
        sim_h = Analyzer(circuit_herald, State([0, 1, 1, 0]))
        results_h = BACKEND.run(sim_h)
        # Then check equivalence of results for all outputs
        for output in results_h.outputs:
            full_state = output[0:2] + State([0, n_output]) + output[2:]
            assert (
                pytest.approx(results_h[State([0, 1, 1, 0]), output])
                == results[State([0, 1, 0, 1, 1, 0]), full_state]
            )

    @pytest.mark.parametrize("n_output", [0, 1, 2])
    def test_herald_equivalance_grouped(self, n_output):
        """
        Checks that heralded and non-heralded circuits produce the same results
        when a grouped circuit is used.
        """
        # Create heralded and versions of same circuit
        unitary = random_unitary(6)
        circuit = Unitary(unitary)
        for i in range(6):
            circuit.loss(i, (i + 1) / 10)
        sub_circuit = Unitary(unitary)
        for i in range(6):
            sub_circuit.loss(i, (i + 1) / 10)
        sub_circuit.herald((2, 2), 0)
        sub_circuit.herald((1, 3), (1, n_output))
        circuit_herald = PhotonicCircuit(4)
        circuit_herald.add(sub_circuit)
        # Simulate both with equivalent inputs
        sim = Analyzer(circuit, State([0, 1, 0, 1, 1, 0]))
        results = BACKEND.run(sim)
        sim_h = Analyzer(circuit_herald, State([0, 1, 1, 0]))
        results_h = BACKEND.run(sim_h)
        # Then check equivalence of results for all outputs
        for output in results_h.outputs:
            full_state = output[0:2] + State([0, n_output]) + output[2:]
            assert (
                pytest.approx(results_h[State([0, 1, 1, 0]), output])
                == results[State([0, 1, 0, 1, 1, 0]), full_state]
            )

    def test_output_heralds_too_large(self):
        """
        Confirms a PhotonNumberError is raised when the number of output heralds
        is larger than the number of photons input into the system.
        """
        circuit = Unitary(random_unitary(4))
        circuit.herald(3, (0, 2))
        sim = Analyzer(circuit, State([1, 0, 0]))
        with pytest.raises(PhotonNumberError):
            sim._generate_task()
