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
    Circuit,
    Parameter,
    State,
    Unitary,
    db_loss_to_decimal,
    random_unitary,
)
from lightworks.emulator import ModeMismatchError, QuickSampler, Sampler


class TestQuickSampler:
    """
    Unit tests to check results produced by QuickSampler object in the
    emulator.
    """

    def test_hom(self):
        """
        Checks sampling a basic 2 photon input onto a 50:50 beam splitter,
        which should undergo HOM, producing outputs of |2,0> and |0,2>.
        """
        circuit = Circuit(2)
        circuit.bs(0)
        sampler = QuickSampler(circuit, State([1, 1]))
        n_sample = 100000
        results = sampler.sample_N_outputs(n_sample, seed=21)
        assert len(results) == 2
        assert 0.49 < results[State([2, 0])] / n_sample < 0.51
        assert 0.49 < results[State([0, 2])] / n_sample < 0.51

    def test_equivalence(self):
        """
        Confirms that the Sampler and QuickSampler produce identical results
        in situations where the QuickSampler assumptions hold true.
        """
        circuit = Unitary(random_unitary(4))
        sampler = Sampler(circuit, State([1, 0, 1, 0]))
        p1 = sampler.probability_distribution
        q_sampler = QuickSampler(circuit, State([1, 0, 1, 0]))
        p2 = q_sampler.probability_distribution
        # Loop through distributions and check they are equal to a reasonable
        # accuracy
        for s1 in p1:
            if s1 not in p2:  # Cannot be identical if a state is missed
                pytest.fail("Missing state in QuickSampler distribution.")
            if round(p1[s1], 8) != round(p2[s1], 8):  # Checks equivalence
                pytest.fail("Probabilities not equivalent.")

    def test_known_result(self):
        """
        Builds a circuit which produces a known result and checks this is found
        at the output.
        """
        # Build circuit
        circuit = Circuit(4)
        circuit.bs(1)
        circuit.mode_swaps({0: 1, 1: 0, 2: 3, 3: 2})
        circuit.bs(0, 3)
        # And check output counts
        sampler = QuickSampler(circuit, State([1, 0, 0, 1]))
        results = sampler.sample_N_outputs(1000)
        assert results[State([0, 1, 1, 0])] == 1000

    def test_sampling(self):
        """
        Checks that the probability distribution calculated by the sampler is
        correct.
        """
        unitary = Unitary(random_unitary(4, seed=43))
        sampler = QuickSampler(
            unitary,
            State([1, 0, 1, 0]),
            photon_counting=False,
            post_select=lambda s: s[0] == 0,
        )
        p = sampler.probability_distribution[State([0, 1, 1, 0])]
        assert p == pytest.approx(0.3156177858, 1e-8)

    def test_sampling_2photons_in_mode(self):
        """
        Checks that the probability distribution calculated by the sampler is
        correct when using 2 photons in a single mode.
        """
        unitary = Unitary(random_unitary(4, seed=43))
        sampler = QuickSampler(
            unitary,
            State([0, 2, 0, 0]),
            photon_counting=False,
            post_select=lambda s: s[0] == 0,
        )
        p = sampler.probability_distribution[State([0, 1, 1, 0])]
        assert p == pytest.approx(0.071330233065, 1e-8)

    def test_lossy_sampling(self):
        """
        Checks that the probability distribution calculated by the sampler with
        a lossy circuit is correct.
        """
        circuit = Circuit(4)
        circuit.bs(0, loss=db_loss_to_decimal(1.3))
        circuit.bs(2, loss=db_loss_to_decimal(2))
        circuit.ps(1, 0.7, loss=db_loss_to_decimal(0.5))
        circuit.ps(3, 0.6, loss=db_loss_to_decimal(0.5))
        circuit.bs(1, loss=db_loss_to_decimal(1.3))
        circuit.bs(2, loss=db_loss_to_decimal(2))
        circuit.ps(1, 0.5, loss=db_loss_to_decimal(0.5))
        sampler = QuickSampler(
            circuit,
            State([1, 0, 1, 0]),
            photon_counting=False,
            post_select=lambda s: s[0] == 0,
        )
        p = sampler.probability_distribution[State([0, 1, 1, 0])]
        assert p == pytest.approx(0.386272843449, 1e-8)

    def test_circuit_update_with_sampler(self):
        """
        Checks that when a circuit is modified then the sampler recalculates
        the probability distribution.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1, 0, 1, 0]))
        p1 = sampler.probability_distribution
        circuit.bs(0)
        circuit.bs(2)
        p2 = sampler.probability_distribution
        assert p1 != p2

    def test_circuit_parameter_update_with_sampler(self):
        """
        Checks that when the parameters of a circuit are updated then the
        corresponding probability distribution is modified.
        """
        p = Parameter(0.3)
        circuit = Circuit(4)
        circuit.bs(0, reflectivity=p)
        circuit.bs(2, reflectivity=p)
        circuit.bs(1, reflectivity=p)
        sampler = QuickSampler(circuit, State([1, 0, 1, 0]))
        p1 = sampler.probability_distribution
        p.set(0.7)
        p2 = sampler.probability_distribution
        assert p1 != p2

    def test_input_update_with_sampler(self):
        """
        Confirms that changing the input state to the sampler alters the
        produced results.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1, 0, 1, 0]))
        p1 = sampler.probability_distribution
        sampler.input_state = State([0, 1, 0, 1])
        p2 = sampler.probability_distribution
        assert p1 != p2

    def test_photon_counting_update_with_sampler(self):
        """
        Confirms that changing the photon_counting attribute changes the
        produced results.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(
            circuit, State([1, 0, 1, 0]), photon_counting=True
        )
        p1 = sampler.probability_distribution
        sampler.photon_counting = False
        p2 = sampler.probability_distribution
        assert p1 != p2

    def test_post_select_update_with_sampler(self):
        """
        Confirms that changing the post_select function changes the produced
        results.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(
            circuit, State([1, 0, 1, 0]), post_select=lambda s: s[0] == 1
        )
        p1 = sampler.probability_distribution
        sampler.post_select = lambda s: s[1] == 1
        p2 = sampler.probability_distribution
        assert p1 != p2

    def test_circuit_assignment(self):
        """
        Confirms that a Circuit cannot be replaced with a non-Circuit through
        the circuit attribute.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1, 0, 1, 0]))
        with pytest.raises(TypeError):
            sampler.circuit = random_unitary(4)

    def test_input_assignmnet(self):
        """
        Checks that the input state of the sampler cannot be assigned to a
        non-State value and requires the correct number of modes.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1, 0, 1, 0]))
        # Incorrect type
        with pytest.raises(TypeError):
            sampler.input_state = [1, 2, 3, 4]
        # Incorrect number of modes
        with pytest.raises(ModeMismatchError):
            sampler.input_state = State([1, 2, 3])

    def test_post_select_assignment(self):
        """
        Confirms that the post_select attribute cannot be replaced with a
        non-function value.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1, 0, 1, 0]))
        with pytest.raises(TypeError):
            sampler.post_select = True

    def test_photon_counting_assignment(self):
        """
        Confirms that the photon_counting attribute cannot be replaced with a
        non-boolean value.
        """
        circuit = Unitary(random_unitary(4))
        sampler = QuickSampler(circuit, State([1, 0, 1, 0]))
        with pytest.raises(TypeError):
            sampler.photon_counting = 1

    def test_herald_equivalent(self):
        """
        Checks that results are equivalent if a herald is used vs
        post-selection on a non-heralded circuit.
        """
        # First calculate distribution without heralding
        circuit = Unitary(random_unitary(6))
        sampler = QuickSampler(
            circuit,
            State([1, 1, 0, 0, 0, 1]),
            post_select=lambda s: s[3] == 1 and s[2] == 0,
        )
        p1 = sampler.probability_distribution
        # Then find with heralding
        circuit.herald(1, 0, 3)
        circuit.herald(0, 2)
        sampler = QuickSampler(circuit, State([1, 0, 0, 1]))
        p2 = sampler.probability_distribution
        for s in p2:
            full_state = s[0:2] + State([0, 1]) + s[2:]
            assert pytest.approx(p2[s]) == p1[full_state]

    def test_herald_equivalent_lossy(self):
        """
        Checks that results are equivalent if a herald is used vs
        post-selection on a non-heralded circuit when lossy modes are
        introduced.
        """
        # First calculate distribution without heralding
        circuit = Unitary(random_unitary(6))
        for i in range(6):
            circuit.loss(i, (i + 1) / 20)
        sampler = QuickSampler(
            circuit,
            State([1, 1, 0, 0, 0, 1]),
            post_select=lambda s: s[3] == 1 and s[2] == 0,
        )
        p1 = sampler.probability_distribution
        # Then find with heralding
        circuit.herald(1, 0, 3)
        circuit.herald(0, 2)
        sampler = QuickSampler(circuit, State([1, 0, 0, 1]))
        p2 = sampler.probability_distribution
        for s in p2:
            full_state = s[0:2] + State([0, 1]) + s[2:]
            assert pytest.approx(p2[s]) == p1[full_state]

    def test_herald_equivalent_grouped(self):
        """
        Checks that results are equivalent if a herald is used vs
        post-selection on a non-heralded circuit when the heralds are featured
        as part of a grouped sub-circuit.
        """
        # First calculate distribution without heralding
        circuit = Unitary(random_unitary(6))
        for i in range(6):
            circuit.loss(i, (i + 1) / 20)
        sampler = QuickSampler(
            circuit,
            State([1, 1, 0, 0, 0, 1]),
            post_select=lambda s: s[3] == 1 and s[2] == 0,
        )
        p1 = sampler.probability_distribution
        # Then find with heralding
        circuit.herald(1, 0, 3)
        circuit.herald(0, 2)
        # Create empty circuit and add original circuit to this
        new_circuit = Circuit(4)
        new_circuit.add(circuit, 0)
        circuit = new_circuit
        sampler = QuickSampler(circuit, State([1, 0, 0, 1]))
        p2 = sampler.probability_distribution
        for s in p2:
            full_state = s[0:2] + State([0, 1]) + s[2:]
            assert pytest.approx(p2[s]) == p1[full_state]

    def test_loss_variable_value(self):
        """
        Checks that QuickSampler is able to support number of required loss
        elements changing if these are part of a parameterized circuits.
        """
        loss = Parameter(0)
        circuit = Circuit(4)
        circuit.bs(0, loss=loss)
        circuit.bs(2, loss=loss)
        circuit.bs(1, loss=loss)
        # Initially sample
        sampler = QuickSampler(circuit, State([1, 0, 1, 0]))
        sampler.sample_N_outputs(10000)
        # Add loss and resample
        loss.set(0.6)
        sampler.sample_N_outputs(10000)
