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

import pytest

from lightworks import (
    ModeMismatchError,
    Parameter,
    PhotonicCircuit,
    PhotonNumberError,
    Simulator,
    State,
    Unitary,
    db_loss_to_decimal,
    random_unitary,
)
from lightworks.emulator import Backend

BACKEND = Backend("permanent")


class TestSimulator:
    """
    Unit tests to check results returned from simulator in different cases,
    including when fermionic statistics are used.
    """

    def test_hom(self):
        """
        Checks the basic 2 mode hom case and confirms the probability of the
        |0,2> state is 0.5.
        """
        circ = PhotonicCircuit(2)
        circ.bs(0)
        sim = Simulator(circ, State([1, 1]), State([2, 0]))
        results = BACKEND.run(sim)
        assert abs(results.array[0, 0]) ** 2 == pytest.approx(0.5, 1e-8)

    def test_single_photon_case(self):
        """
        Runs a single photon sim and checks the calculated unitary matches the
        target unitary.
        """
        n_modes = 4
        unitary = random_unitary(n_modes)
        unitary_circ = Unitary(unitary)
        states = []
        for i in range(n_modes):
            states.append(State([int(i == j) for j in range(n_modes)]))
        sim = Simulator(unitary_circ, states, states)
        results = BACKEND.run(sim)
        assert (unitary.T.round(8) == results.array.round(8)).all()

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
        # And check output probability
        sim = Simulator(circuit, State([1, 0, 0, 1]), State([0, 1, 1, 0]))
        results = BACKEND.run(sim)
        assert abs(results.array[0, 0]) ** 2 == pytest.approx(0.5, 1e-8)

    def test_multi_photon_case(self):
        """
        Runs a multi-photon sim and checks the correct value is found for one
        input/output.
        """
        unitary = random_unitary(4, seed=10)
        unitary = Unitary(unitary)
        sim = Simulator(unitary, State([1, 0, 1, 0]), State([0, 2, 0, 0]))
        results = BACKEND.run(sim)
        x = results.array[0, 0]
        assert x == pytest.approx(
            -0.18218877232689196 - 0.266230290128261j, 1e-8
        )

    def test_multi_photon_output_not_specified_case(self):
        """
        Runs a multi-photon sim and checks the correct value is found for one
        input with outputs not specified.
        """
        unitary = random_unitary(4, seed=10)
        unitary = Unitary(unitary)
        sim = Simulator(unitary, State([1, 0, 1, 0]))
        results = BACKEND.run(sim)
        x = results[State([1, 0, 1, 0]), State([0, 2, 0, 0])]
        assert x == pytest.approx(
            -0.18218877232689196 - 0.266230290128261j, 1e-8
        )

    def test_lossy_multi_photon_case(self):
        """
        Runs a lossy multi-photon sim and checks the correct value is found for
        one input/output.
        """
        circ = PhotonicCircuit(4)
        circ.bs(0, loss=db_loss_to_decimal(2))
        circ.ps(1, phi=0.3)
        circ.bs(1, loss=db_loss_to_decimal(2))
        circ.bs(2, loss=db_loss_to_decimal(2))
        circ.ps(1, phi=0.5)
        circ.bs(1, loss=db_loss_to_decimal(2))
        sim = Simulator(circ, State([2, 0, 0, 0]), State([0, 1, 1, 0]))
        results = BACKEND.run(sim)
        x = results.array[0, 0]
        assert x == pytest.approx(
            0.03647550871283556 + 0.01285838825922496j, 1e-8
        )

    def test_circuit_update(self):
        """Used to check circuit updates affect simulator results."""
        unitary = Unitary(random_unitary(4))
        # Create simulator and get initial results
        sim = Simulator(unitary, State([1, 0, 1, 0]), State([0, 2, 0, 0]))
        results = BACKEND.run(sim)
        x = results.array[0, 0]
        # Update circuit adn re-simulate
        unitary.bs(0)
        results = BACKEND.run(sim)
        x2 = results.array[0, 0]
        assert x != x2

    def test_circuit_parameter_update(self):
        """
        Used to check circuit updates through parameter changes affect
        simulator results.
        """
        param = Parameter(0.3)
        circuit = PhotonicCircuit(4)
        circuit.bs(0, reflectivity=param)
        circuit.bs(2, reflectivity=param)
        circuit.bs(1, reflectivity=param)
        # Create simulator and get initial results
        sim = Simulator(circuit, State([1, 0, 1, 0]), State([0, 2, 0, 0]))
        results = BACKEND.run(sim)
        x = results.array[0, 0]
        # Update parameter and re-simulate
        param.set(0.65)
        results = BACKEND.run(sim)
        x2 = results.array[0, 0]
        assert x != x2

    def test_circuit_assignment(self):
        """
        Checks that an incorrect value cannot be assigned to the circuit
        attribute.
        """
        circuit = Unitary(random_unitary(4))
        sim = Simulator(circuit, State([1, 0, 1, 0]))
        with pytest.raises(TypeError):
            sim.circuit = random_unitary(5)

    def test_varied_input_n_raises_error(self):
        """
        Checks that an error is raised if it attempted to use inputs with
        different photon numbers.
        """
        # Create circuit and simulator object
        circuit = Unitary(random_unitary(4))
        # Without output specified
        sim = Simulator(circuit, [State([1, 0, 1, 0]), State([0, 1, 0, 0])])
        with pytest.raises(PhotonNumberError):
            BACKEND.run(sim)
        # With some outputs specified
        sim = Simulator(
            circuit,
            [State([1, 0, 1, 0]), State([0, 1, 0, 0])],
            [State([1, 0, 1, 0]), State([0, 1, 0, 1])],
        )
        with pytest.raises(PhotonNumberError):
            BACKEND.run(sim)

    def test_varied_output_n_raises_error(self):
        """
        Checks that an error is raised if it attempted to use outputs with
        different photon numbers to the input or each other.
        """
        # Create circuit and simulator object
        circuit = Unitary(random_unitary(4))
        # With different number to input
        sim = Simulator(
            circuit,
            [State([1, 0, 1, 0]), State([0, 1, 0, 1])],
            [State([0, 0, 1, 0]), State([0, 1, 0, 0])],
        )
        with pytest.raises(PhotonNumberError):
            BACKEND.run(sim)
        # With different number to each other
        sim = Simulator(
            circuit,
            [State([1, 0, 1, 0]), State([0, 1, 0, 1])],
            [State([1, 0, 1, 0]), State([0, 1, 0, 0])],
        )
        with pytest.raises(PhotonNumberError):
            BACKEND.run(sim)

    def test_incorrect_input_length(self):
        """
        Confirms an error is raised when an input which does not match the
        required circuit modes is supplied.
        """
        # Create circuit and simulator object
        circuit = Unitary(random_unitary(4))
        # Attempt to simulate with input which is too short
        with pytest.raises(ModeMismatchError):
            Simulator(circuit, State([1, 0, 1]))
        # And then which is too long
        with pytest.raises(ModeMismatchError):
            Simulator(circuit, State([1, 0, 1, 0, 1]))

    def test_incorrect_input_length_herald(self):
        """
        Confirms an error is raised when an input which does not match the
        required circuit modes is supplied, when heralding is used in the
        original circuit.
        """
        # Create circuit and simulator object
        circuit = Unitary(random_unitary(4))
        circuit.herald(1, 0, 2)
        # Attempt to simulate with input which is too short
        with pytest.raises(ModeMismatchError):
            Simulator(circuit, State([1, 0]))
        # And then which is too long
        with pytest.raises(ModeMismatchError):
            Simulator(circuit, State([1, 0, 1, 0]))

    def test_incorrect_input_length_herald_grouped(self):
        """
        Confirms an error is raised when an input which does not match the
        required circuit modes is supplied, when heralding is used in a
        sub-circuit of the original circuit.
        """
        # Create circuit and simulator object
        circuit = Unitary(random_unitary(4))
        sub_circuit = Unitary(random_unitary(4))
        sub_circuit.herald(1, 0, 2)
        circuit.add(sub_circuit, 1)
        # Attempt to simulate with input which is too short
        with pytest.raises(ModeMismatchError):
            Simulator(circuit, State([1, 0, 1]))
        # And then which is too long
        with pytest.raises(ModeMismatchError):
            Simulator(circuit, State([1, 0, 1, 0, 0]))

    def test_herald_not_herald_equivalance(self):
        """
        Checks that the results from the Simulator are equivalent when using a
        heralded circuit and getting the same outputs from a non-heralded
        circuit.
        """
        # Create heralded and versions of same circuit
        unitary = random_unitary(6)
        circuit = Unitary(unitary)
        circuit_herald = Unitary(unitary)
        circuit_herald.herald(0, 2, 2)
        circuit_herald.herald(1, 1, 3)
        # Simulate both with equivalent inputs
        sim = Simulator(circuit, State([0, 1, 0, 1, 1, 0]))
        results = BACKEND.run(sim)
        sim_h = Simulator(circuit_herald, State([0, 1, 1, 0]))
        results_h = BACKEND.run(sim_h)
        # Then check equivalence of results for all outputs
        for output in results_h.outputs:
            full_state = output[0:2] + State([0, 1]) + output[2:]
            assert (
                pytest.approx(results_h[State([0, 1, 1, 0]), output])
                == results[State([0, 1, 0, 1, 1, 0]), full_state]
            )

    def test_herald_not_herald_equivalance_lossy(self):
        """
        Checks that the results from the Simulator are equivalent when using a
        heralded circuit and getting the same outputs from a non-heralded
        circuit, while including loss modes.
        """
        # Create heralded and versions of same circuit
        unitary = random_unitary(6)
        circuit = Unitary(unitary)
        for i in range(6):
            circuit.loss(i, (i + 1) / 10)
        circuit_herald = Unitary(unitary)
        for i in range(6):
            circuit_herald.loss(i, (i + 1) / 10)
        circuit_herald.herald(0, 2, 2)
        circuit_herald.herald(1, 1, 3)
        # Simulate both with equivalent inputs
        sim = Simulator(circuit, State([0, 1, 0, 1, 1, 0]))
        results = BACKEND.run(sim)
        sim_h = Simulator(circuit_herald, State([0, 1, 1, 0]))
        results_h = BACKEND.run(sim_h)
        # Then check equivalence of results for all outputs
        for output in results_h.outputs:
            full_state = output[0:2] + State([0, 1]) + output[2:]
            assert (
                pytest.approx(results_h[State([0, 1, 1, 0]), output])
                == results[State([0, 1, 0, 1, 1, 0]), full_state]
            )

    def test_herald_not_herald_equivalance_grouped(self):
        """
        Checks that the results from the Simulator are equivalent when using a
        heralded circuit and getting the same outputs from a non-heralded
        circuit, while the heralded circuit contains heralds within a grouped
        sub-circuit.
        """
        # Create heralded and versions of same circuit
        unitary = random_unitary(6)
        circuit = Unitary(unitary)
        for i in range(6):
            circuit.loss(i, (i + 1) / 10)
        sub_circuit = Unitary(unitary)
        for i in range(6):
            sub_circuit.loss(i, (i + 1) / 10)
        sub_circuit.herald(0, 2, 2)
        sub_circuit.herald(1, 1, 3)
        circuit_herald = PhotonicCircuit(4)
        circuit_herald.add(sub_circuit)
        # Simulate both with equivalent inputs
        sim = Simulator(circuit, State([0, 1, 0, 1, 1, 0]))
        results = BACKEND.run(sim)
        sim_h = Simulator(circuit_herald, State([0, 1, 1, 0]))
        results_h = BACKEND.run(sim_h)
        # Then check equivalence of results for all outputs
        for output in results_h.outputs:
            full_state = output[0:2] + State([0, 1]) + output[2:]
            assert (
                pytest.approx(results_h[State([0, 1, 1, 0]), output])
                == results[State([0, 1, 0, 1, 1, 0]), full_state]
            )
