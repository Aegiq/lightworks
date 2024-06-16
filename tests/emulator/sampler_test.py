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

from lightworks import State, Unitary, Circuit, random_unitary, Parameter
from lightworks.emulator import ModeMismatchError
from lightworks.emulator import Sampler, Source, Detector, Backend

import pytest


class TestSamplerGeneral:
    """
    Unit tests to check non-backend specific functionality of Sampler.
    """

    def test_sample_n_states_seed(self):
        """
        Checks that two successive function calls with a consistent seed
        produce the same result.
        """
        circuit = Unitary(random_unitary(4))
        sampler = Sampler(circuit, State([1, 0, 1, 0]))
        results = sampler.sample_N_inputs(5000, seed=1)
        results2 = sampler.sample_N_inputs(5000, seed=1)
        assert results.dictionary == results2.dictionary

    def test_sample_n_states_seed_detector(self):
        """
        Checks that two successive function calls with a consistent seed
        produce the same result when an imperfect detector is used.
        """
        circuit = Unitary(random_unitary(4))
        sampler = Sampler(
            circuit,
            State([1, 0, 1, 0]),
            detector=Detector(efficiency=0.5, p_dark=1e-3),
        )
        results = sampler.sample_N_inputs(5000, seed=1)
        results2 = sampler.sample_N_inputs(5000, seed=1)
        assert results.dictionary == results2.dictionary

    def test_circuit_update_with_sampler(self):
        """
        Checks that when a circuit is modified then the sampler recalculates
        the probability distribution.
        """
        circuit = Unitary(random_unitary(4))
        sampler = Sampler(circuit, State([1, 0, 1, 0]))
        p1 = sampler.probability_distribution
        circuit.add_bs(0)
        circuit.add_bs(2)
        p2 = sampler.probability_distribution
        assert p1 != p2

    def test_circuit_parameter_update_with_sampler(self):
        """
        Checks that when the parameters of a circuit are updated then the
        corresponding probability distribution is modified.
        """
        p = Parameter(0.3)
        circuit = Circuit(4)
        circuit.add_bs(0, reflectivity=p)
        circuit.add_bs(2, reflectivity=p)
        circuit.add_bs(1, reflectivity=p)
        sampler = Sampler(circuit, State([1, 0, 1, 0]))
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
        sampler = Sampler(circuit, State([1, 0, 1, 0]))
        p1 = sampler.probability_distribution
        sampler.input_state = State([0, 1, 0, 1])
        p2 = sampler.probability_distribution
        assert p1 != p2

    def test_circuit_assignment(self):
        """
        Confirms that a Circuit cannot be replaced with a non-Circuit through
        the circuit attribute.
        """
        circuit = Unitary(random_unitary(4))
        sampler = Sampler(circuit, State([1, 0, 1, 0]))
        with pytest.raises(TypeError):
            sampler.circuit = random_unitary(4)

    def test_input_assignmnet(self):
        """
        Checks that the input state of the sampler cannot be assigned to a
        non-State value and requires the correct number of modes.
        """
        circuit = Unitary(random_unitary(4))
        sampler = Sampler(circuit, State([1, 0, 1, 0]))
        # Incorrect type
        with pytest.raises(TypeError):
            sampler.input_state = [1, 2, 3, 4]
        # Incorrect number of modes
        with pytest.raises(ModeMismatchError):
            sampler.input_state = State([1, 2, 3])

    def test_source_assignment(self):
        """
        Confirms that a Source cannot be replaced with a non-source object
        through the source attribute.
        """
        circuit = Unitary(random_unitary(4))
        sampler = Sampler(circuit, State([1, 0, 1, 0]))
        with pytest.raises(TypeError):
            sampler.source = random_unitary(4)

    def test_detector_assignment(self):
        """
        Confirms that a Detector cannot be replaced with a non-detector object
        through the detector attribute.
        """
        circuit = Unitary(random_unitary(4))
        sampler = Sampler(circuit, State([1, 0, 1, 0]))
        with pytest.raises(TypeError):
            sampler.detector = random_unitary(4)

    def test_imperfect_source_update_with_sampler(self):
        """
        Checks that updating the parameters of a source will alter the
        calculated probability distribution from the sampler.
        """
        circuit = Unitary(random_unitary(4))
        source = Source(
            brightness=0.9,
            purity=0.9,
            indistinguishability=0.9,
            probability_threshold=1e-6,
        )
        sampler = Sampler(circuit, State([1, 0, 1, 0]), source=source)
        p1 = sampler.probability_distribution
        # Indistinguishability
        source.indistinguishability = 0.2
        p2 = sampler.probability_distribution
        assert p1 != p2
        # Purity (reset previous variable to original value)
        source.indistinguishability = 0.9
        source.purity = 0.7
        p2 = sampler.probability_distribution
        assert p1 != p2
        # Brightness
        source.purity = 0.9
        source.brightness = 0.4
        p2 = sampler.probability_distribution
        assert p1 != p2
        # Probability threshold
        source.brightness = 0.9
        source.probability_threshold = 1e-3
        p2 = sampler.probability_distribution
        assert p1 != p2
        # Return all values to defaults and check this then returns the
        # original distribution
        source.probability_threshold = 1e-6
        p2 = sampler.probability_distribution
        assert p1 == p2

    def test_slos_equivalence_basic(self):
        """
        Checks probability distribution calculation from a simple unitary is
        nearly equivalent using both permanent and slos calculations.
        """
        circuit = Unitary(random_unitary(4))
        # Permanent
        sampler = Sampler(circuit, State([1, 0, 1, 0]), backend="permanent")
        p1 = sampler.probability_distribution
        # SLOS
        sampler.backend = "slos"
        p2 = sampler.probability_distribution
        for s in p1:
            if round(p1[s], 8) != round(p2[s], 8):
                pytest.fail("Methods do not produce equivalent distributions.")

    def test_slos_equivalence_complex(self):
        """
        Checks probability distribution calculation is nearly equivalent using
        both permanent and slos calculations, when using loss and an imperfect
        source.
        """
        circuit = Unitary(random_unitary(4))
        for i in range(4):
            circuit.add_loss(i, 1)
        source = Source(indistinguishability=0.9, brightness=0.9, purity=0.9)
        # Permanent
        sampler = Sampler(
            circuit, State([1, 0, 1, 0]), source=source, backend="permanent"
        )
        p1 = sampler.probability_distribution
        # SLOS
        sampler.backend = "slos"
        p2 = sampler.probability_distribution
        # Test equivalence
        for s in p1:
            if round(p1[s], 8) != round(p2[s], 8):
                pytest.fail("Methods do not produce equivalent distributions.")

    def test_backend_updated_recalculates(self):
        """
        Checks that updating the backend causes recalculation of the
        probability distribution.
        """
        circuit = Unitary(random_unitary(4))
        # Get initial distribution
        sampler = Sampler(circuit, State([1, 0, 1, 0]), backend="permanent")
        sampler.probability_distribution  # noqa: B018
        # Then switch to SLOS
        sampler.backend = "slos"
        # Check method below returns True
        assert sampler._check_parameter_updates()

    def test_backend_assignment(self):
        """
        Tests that backend can be set both by assigning backend object and by
        providing a string.
        """
        sampler = Sampler(Circuit(4), State([1, 0, 1, 0]))
        # Assign object directly
        sampler.backend = Backend("permanent")
        # Assign string
        sampler.backend = "permanent"
        assert isinstance(sampler.backend, Backend)


@pytest.mark.parametrize("backend", [Backend("permanent"), Backend("slos")])
class TestSamplerCalculationBackends:
    """
    Unit tests to check results produced by Sampler object in the emulator with
    both SLOS and permanent backends.
    """

    def test_hom(self, backend):
        """
        Checks sampling a basic 2 photon input onto a 50:50 beam splitter,
        which should undergo HOM, producing outputs of |2,0> and |0,2>.
        """
        circuit = Circuit(2)
        circuit.add_bs(0)
        sampler = Sampler(circuit, State([1, 1]), backend=backend)
        n_sample = 100000
        results = sampler.sample_N_inputs(n_sample, seed=21)
        assert len(results) == 2
        assert 0.49 < results[State([2, 0])] / n_sample < 0.51
        assert 0.49 < results[State([0, 2])] / n_sample < 0.51

    def test_hom_sample_n_outputs(self, backend):
        """
        Checks a lossy hom experiment with sample N outputs produces outputs of
        |2,0> and |0,2>.
        """
        circuit = Circuit(2)
        circuit.add_bs(0, loss=1)
        sampler = Sampler(circuit, State([1, 1]), backend=backend)
        n_sample = 100000
        results = sampler.sample_N_outputs(n_sample, seed=54, min_detection=2)
        assert sum(results.values()) == n_sample
        assert 0.49 < results[State([2, 0])] / n_sample < 0.51
        assert 0.49 < results[State([0, 2])] / n_sample < 0.51

    def test_known_result(self, backend):
        """
        Builds a circuit which produces a known result and checks this is found
        at the output.
        """
        # Build circuit
        circuit = Circuit(4)
        circuit.add_bs(1)
        circuit.add_mode_swaps({0: 1, 1: 0, 2: 3, 3: 2})
        circuit.add_bs(0, 3)
        # And check output counts
        sampler = Sampler(circuit, State([1, 0, 0, 1]), backend=backend)
        results = sampler.sample_N_inputs(1000)
        assert results[State([0, 1, 1, 0])] == 1000

    def test_known_result_single_sample(self, backend):
        """
        Builds a circuit which produces a known result and checks this is found
        at the output, when using the sample method to get a single value.
        """
        # Build circuit
        circuit = Circuit(4)
        circuit.add_bs(1)
        circuit.add_mode_swaps({0: 1, 1: 0, 2: 3, 3: 2})
        circuit.add_bs(0, 3)
        # And check output counts
        sampler = Sampler(circuit, State([1, 0, 0, 1]), backend=backend)
        output = sampler.sample()
        assert output == State([0, 1, 1, 0])

    def test_sampling_perfect_source(self, backend):
        """
        Checks that the probability distribution calculated by the sampler is
        correct for a perfect source.
        """
        unitary = Unitary(random_unitary(4, seed=20))
        sampler = Sampler(unitary, State([1, 0, 1, 0]), backend=backend)
        p = sampler.probability_distribution[State([0, 1, 1, 0])]
        assert p == pytest.approx(0.112093500, 1e-8)

    def test_sampling_imperfect_source(self, backend):
        """
        Checks that the probability distribution calculated by the sampler is
        correct for an imperfect source.
        """
        unitary = Unitary(random_unitary(4, seed=20))
        source = Source(purity=0.9, brightness=0.9, indistinguishability=0.9)
        sampler = Sampler(
            unitary, State([1, 0, 1, 0]), source=source, backend=backend
        )
        p = sampler.probability_distribution[State([0, 0, 1, 0])]
        assert p == pytest.approx(0.0129992654, 1e-8)

    def test_sampling_2photons_in_mode_perfect_source(self, backend):
        """
        Checks that the probability distribution calculated by the sampler is
        correct for a perfect source with 2 photons in single input mode.
        """
        unitary = Unitary(random_unitary(4, seed=20))
        sampler = Sampler(unitary, State([0, 2, 0, 0]), backend=backend)
        p = sampler.probability_distribution[State([0, 1, 1, 0])]
        assert p == pytest.approx(0.2875114938, 1e-8)

    def test_sampling_2photons_in_mode_imperfect_source(self, backend):
        """
        Checks that the probability distribution calculated by the sampler is
        correct for an imperfect source with 2 photons in single input mode.
        """
        unitary = Unitary(random_unitary(4, seed=20))
        source = Source(purity=0.9, brightness=0.9, indistinguishability=0.9)
        sampler = Sampler(
            unitary, State([0, 2, 0, 0]), source=source, backend=backend
        )
        p = sampler.probability_distribution[State([0, 0, 1, 0])]
        assert p == pytest.approx(0.09767722765, 1e-8)

    def test_lossy_sampling_perfect_source(self, backend):
        """
        Checks that the probability distribution calculated by the sampler with
        a lossy circuit is correct for a perfect source.
        """
        # Build circuit
        circuit = Circuit(4)
        circuit.add_bs(0, loss=1)
        circuit.add_bs(2, loss=2)
        circuit.add_ps(1, 0.3, loss=0.5)
        circuit.add_ps(3, 0.3, loss=0.5)
        circuit.add_bs(1, loss=1)
        circuit.add_bs(2, loss=2)
        circuit.add_ps(1, 0.3, loss=0.5)
        # Sample from circuit
        sampler = Sampler(circuit, State([1, 0, 1, 0]), backend=backend)
        p = sampler.probability_distribution[State([0, 1, 1, 0])]
        assert p == pytest.approx(0.01111424631, 1e-8)
        p = sampler.probability_distribution[State([0, 0, 0, 0])]
        assert p == pytest.approx(0.24688532527, 1e-8)

    def test_lossy_sampling_imperfect_source(self, backend):
        """
        Checks that the probability distribution calculated by the sampler with
        a lossy circuit is correct for an imperfect source.
        """
        # Build circuit
        circuit = Circuit(4)
        circuit.add_bs(0, loss=1)
        circuit.add_bs(2, loss=2)
        circuit.add_ps(1, 0.3, loss=0.5)
        circuit.add_ps(3, 0.3, loss=0.5)
        circuit.add_bs(1, loss=1)
        circuit.add_bs(2, loss=2)
        circuit.add_ps(1, 0.3, loss=0.5)
        # Sample from circuit
        source = Source(purity=0.9, brightness=0.9, indistinguishability=0.9)
        sampler = Sampler(
            circuit, State([1, 0, 1, 0]), source=source, backend=backend
        )
        p = sampler.probability_distribution[State([0, 0, 1, 0])]
        assert p == pytest.approx(0.03122592963, 1e-8)
        p = sampler.probability_distribution[State([0, 0, 0, 0])]
        assert p == pytest.approx(0.28709359025, 1e-8)

    def test_imperfect_detection(self, backend):
        """Tests the behaviour of detectors with less than ideal efficiency."""
        circuit = Unitary(random_unitary(4))
        # Control
        detector = Detector(efficiency=1)
        sampler = Sampler(
            circuit, State([1, 0, 1, 0]), detector=detector, backend=backend
        )
        results = sampler.sample_N_inputs(1000)
        undetected_photons = False
        for s in results.keys():
            if s.n_photons < 2:
                undetected_photons = True
                break
        assert not undetected_photons
        # With lossy detector
        detector = Detector(efficiency=0.5)
        sampler = Sampler(
            circuit, State([1, 0, 1, 0]), detector=detector, backend=backend
        )
        results = sampler.sample_N_inputs(1000)
        undetected_photons = False
        for s in results.keys():
            if s.n_photons < 2:
                undetected_photons = True
                break
        assert undetected_photons

    def test_detector_dark_counts(self, backend):
        """Confirms detector dark counts are introduced as expected."""
        circuit = Unitary(random_unitary(4))
        # Control
        detector = Detector(p_dark=0)
        sampler = Sampler(
            circuit, State([0, 0, 0, 0]), detector=detector, backend=backend
        )
        results = sampler.sample_N_inputs(1000)
        dark_counts = False
        for s in results.keys():
            if s.n_photons > 0:
                dark_counts = True
                break
        assert not dark_counts
        # With dark counts enabled
        detector = Detector(p_dark=0.1)
        sampler = Sampler(
            circuit, State([0, 0, 0, 0]), detector=detector, backend=backend
        )
        results = sampler.sample_N_inputs(1000)
        dark_counts = False
        for s in results.keys():
            if s.n_photons > 0:
                dark_counts = True
                break
        assert dark_counts

    def test_detector_photon_counting(self, backend):
        """
        Checks that detector photon counting control alters the output states
        as expected after sampling.
        """
        circuit = Unitary(random_unitary(4))
        # Photon number resolving
        detector = Detector(photon_counting=True)
        sampler = Sampler(
            circuit, State([1, 0, 1, 0]), detector=detector, backend=backend
        )
        results = sampler.sample_N_inputs(1000)
        all_2_photon_states = True
        for s in results.keys():
            if s.n_photons < 2:
                all_2_photon_states = False
                break
        assert all_2_photon_states
        # Non-photon number resolving
        detector = Detector(photon_counting=False)
        sampler = Sampler(
            circuit, State([0, 0, 0, 0]), detector=detector, backend=backend
        )
        results = sampler.sample_N_inputs(1000)
        sub_2_photon_states = False
        for s in results.keys():
            if s.n_photons < 2:
                sub_2_photon_states = True
                break
        assert sub_2_photon_states

    @pytest.mark.flaky(reruns=2)
    def test_herald_equivalent(self, backend):
        """
        Checks that results are equivalent if a herald is used vs
        post-selection on a non-heralded circuit.
        """
        circuit = Unitary(random_unitary(6))
        # Sampler without built-in heralds
        sampler = Sampler(circuit, State([1, 1, 0, 1, 0, 0]), backend=backend)
        results = sampler.sample_N_outputs(
            50000, post_select=lambda s: s[1] == 1 and s[3] == 0
        )
        # Then add and re-sample
        circuit.add_herald(1, 0, 1)
        circuit.add_herald(0, 2, 3)
        sampler = Sampler(circuit, State([1, 1, 0, 0]), backend=backend)
        results2 = sampler.sample_N_outputs(50000)
        # Check all results with outputs greater than 2000
        for s, c in results2.items():
            if c > 2000:
                full_s = s[0:1] + State([1]) + s[1:2] + State([0]) + s[2:]
                # Check results are within 10%
                assert pytest.approx(results[full_s], 0.1) == results2[s]

    @pytest.mark.flaky(reruns=2)
    def test_herald_equivalent_imperfect_source(self, backend):
        """
        Checks that results are equivalent if a herald is used vs
        post-selection on a non-heralded circuit, while also including an
        imperfect photon source.
        """
        circuit = Unitary(random_unitary(6))
        # Define source to use
        source = Source(purity=0.9, brightness=0.9, indistinguishability=0.9)
        # Sampler without built-in heralds
        sampler = Sampler(
            circuit, State([1, 1, 0, 1, 0, 0]), backend=backend, source=source
        )
        results = sampler.sample_N_outputs(
            50000, post_select=lambda s: s[1] == 1 and s[3] == 0
        )
        # Then add and re-sample
        circuit.add_herald(1, 0, 1)
        circuit.add_herald(0, 2, 3)
        sampler = Sampler(
            circuit, State([1, 1, 0, 0]), backend=backend, source=source
        )
        results2 = sampler.sample_N_outputs(50000)
        # Check all results with outputs greater than 2000
        for s, c in results2.items():
            if c > 2000:
                full_s = s[0:1] + State([1]) + s[1:2] + State([0]) + s[2:]
                # Check results are within 10%
                assert pytest.approx(results[full_s], 0.1) == results2[s]

    @pytest.mark.flaky(reruns=2)
    def test_herald_equivalent_lossy(self, backend):
        """
        Checks that results are equivalent if a herald is used vs
        post-selection on a non-heralded lossy circuit.
        """
        circuit = Unitary(random_unitary(6))
        for i in range(6):
            circuit.add_loss(i, i + 1)
        # Sampler without built-in heralds
        sampler = Sampler(circuit, State([1, 1, 0, 1, 0, 0]), backend=backend)
        results = sampler.sample_N_outputs(
            50000, post_select=lambda s: s[1] == 1 and s[3] == 0
        )
        # Then add and re-sample
        circuit.add_herald(1, 0, 1)
        circuit.add_herald(0, 2, 3)
        sampler = Sampler(circuit, State([1, 1, 0, 0]), backend=backend)
        results2 = sampler.sample_N_outputs(50000)
        # Check all results with outputs greater than 2000
        for s, c in results2.items():
            if c > 2000:
                full_s = s[0:1] + State([1]) + s[1:2] + State([0]) + s[2:]
                # Check results are within 10%
                assert pytest.approx(results[full_s], 0.1) == results2[s]

    @pytest.mark.flaky(reruns=2)
    def test_herald_equivalent_lossy_imperfect_source(self, backend):
        """
        Checks that results are equivalent if a herald is used vs
        post-selection on a non-heralded lossy circuit, while also including an
        imperfect photon source.
        """
        circuit = Unitary(random_unitary(6))
        for i in range(6):
            circuit.add_loss(i, i + 1)
        # Define source to use
        source = Source(purity=0.9, brightness=0.9, indistinguishability=0.9)
        # Sampler without built-in heralds
        sampler = Sampler(
            circuit, State([1, 1, 0, 1, 0, 0]), backend=backend, source=source
        )
        results = sampler.sample_N_outputs(
            50000, post_select=lambda s: s[1] == 1 and s[3] == 0
        )
        # Then add and re-sample
        circuit.add_herald(1, 0, 1)
        circuit.add_herald(0, 2, 3)
        sampler = Sampler(
            circuit, State([1, 1, 0, 0]), backend=backend, source=source
        )
        results2 = sampler.sample_N_outputs(50000)
        # Check all results with outputs greater than 2000
        for s, c in results2.items():
            if c > 2000:
                full_s = s[0:1] + State([1]) + s[1:2] + State([0]) + s[2:]
                # Check results are within 10%
                assert pytest.approx(results[full_s], 0.1) == results2[s]

    def test_hom_imperfect_brightness(self, backend):
        """
        Checks sampling a basic 2 photon input onto a 50:50 beam splitter,
        which should undergo HOM, producing outputs of |2,0> and |0,2>.
        Includes imperfect brightness.
        """
        circuit = Circuit(2)
        circuit.add_bs(0)
        sampler = Sampler(
            circuit,
            State([1, 1]),
            backend=backend,
            source=Source(brightness=0.8),
        )
        n_sample = 100000
        results = sampler.sample_N_outputs(n_sample, seed=21, min_detection=2)
        assert len(results) == 2
        assert 0.49 < results[State([2, 0])] / n_sample < 0.51
        assert 0.49 < results[State([0, 2])] / n_sample < 0.51

    def test_loss_variable_value(self, backend):
        """
        Checks that Sampler is able to support number of required loss elements
        changing if these are part of a parameterized circuits.
        """
        loss = Parameter(0)
        circuit = Circuit(4)
        circuit.add_bs(0, loss=loss)
        circuit.add_bs(2, loss=loss)
        circuit.add_bs(1, loss=loss)
        # Initially sample
        sampler = Sampler(circuit, State([1, 0, 1, 0]), backend=backend)
        sampler.sample_N_inputs(10000)
        # Add loss and resample
        loss.set(1)
        sampler.sample_N_inputs(10000)
