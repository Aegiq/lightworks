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

from lightworks import Sampler, Simulator, State, Unitary, random_unitary
from lightworks.emulator import Backend, BackendError
from lightworks.emulator.backends import (
    PermanentBackend,
    SLOSBackend,
)


class TestBackend:
    """
    Unit tests for ensuring backend object remains functioning correctly.
    """

    @pytest.mark.parametrize("backend", ["permanent", "slos"])
    def test_valid_backend(self, backend):
        """Checks all currently valid backends can be set."""
        Backend(backend)

    def test_invalid_backends(self):
        """Checks an invalid backend raises a ValueError"""
        with pytest.raises(ValueError):
            Backend("not_a_backend")

    def test_invalid_backend_for_task(self):
        """
        Checks an error is raised when an invalid backend is chosen for a task.
        """
        backend = Backend("slos")
        sim = Simulator(Unitary(random_unitary(4)), State([1, 0, 1, 0]))
        with pytest.raises(BackendError):
            backend.run(sim)

    @pytest.mark.parametrize("backend_type", ["permanent", "slos"])
    def test_full_probability_distribution(self, backend_type):
        """
        Check against a known result for the full probability distribution
        while using the both permanent and slos backends.
        """
        backend = Backend(backend_type)._Backend__backend
        unitary = random_unitary(3, seed=2)
        dist = backend.full_probability_distribution(
            Unitary(unitary)._build(), State([1, 1, 0])
        )
        known_dist = {
            State([2, 0, 0]): 0.2600968016733883,
            State([1, 1, 0]): 0.22397644081164791,
            State([0, 2, 0]): 0.13677578896115924,
            State([1, 0, 1]): 0.04763948802896683,
            State([0, 1, 1]): 0.2377381500275615,
            State([0, 0, 2]): 0.09377333049727585,
        }
        for d in dist:
            if d not in known_dist:
                pytest.fail(
                    "State from produced distribution not in known "
                    "distribution."
                )
            assert known_dist[d] == pytest.approx(dist[d], 1e-8)

    def test_permanent_slos_equivalence(self):
        """
        Check permanent and slos backends produce equivalent probability
        distributions.
        """
        circuit = Unitary(random_unitary(6))
        input_state = State([1, 0, 1, 0, 1, 0])
        # Find distributions
        backend_p = PermanentBackend()
        p1 = backend_p.full_probability_distribution(
            circuit._build(), input_state
        )
        backend_s = SLOSBackend()
        p2 = backend_s.full_probability_distribution(
            circuit._build(), input_state
        )
        # Check equivalence
        for s in p1:
            if round(p1[s], 8) != round(p2[s], 8):
                pytest.fail("Methods do not produce equivalent distributions.")

    def test_backend_str_return(self):
        """
        Check that backend value is stored and returned correctly when using
        the str operator.
        """
        backend = Backend("permanent")
        assert str(backend) == "permanent"

    def test_backend_repr_return(self):
        """
        Checks that backend value is correctly included in __repr__ for
        backend.
        """
        backend = Backend("permanent")
        assert "permanent" in repr(backend)


class TestPermanent:
    """
    Specific functions for testing Permanent calculation remains functional
    and consistent.
    """

    def test_single_photon_return(self):
        """
        Confirm elements of unitary are returned by permanent in single photon
        cases.
        """
        unitary = random_unitary(4)
        # Diagonal
        assert unitary[0, 0] == PermanentBackend().probability_amplitude(
            unitary, State([1, 0, 0, 0]), State([1, 0, 0, 0])
        )
        # Off-diagonal
        assert unitary[2, 1] == PermanentBackend().probability_amplitude(
            unitary, State([0, 1, 0, 0]), State([0, 0, 1, 0])
        )

    def test_known_result(self):
        """
        Calculate a permanent value and check it matches a previously
        calculated result.
        """
        unitary = random_unitary(6, seed=23)
        r = PermanentBackend().probability_amplitude(
            unitary, State([1, 0, 1, 0, 1, 0]), State([0, 1, 1, 0, 0, 1])
        )
        assert r == pytest.approx(-0.02042658999324299 - 0.02226528732909283j)

    def test_probability_amplitude(self):
        """
        Confirms that expected elements from the unitary matrix are returned
        for a single photon input.
        """
        backend = PermanentBackend()
        unitary = random_unitary(4)
        # Diagonal
        assert unitary[0, 0] == backend.probability_amplitude(
            unitary, [1, 0, 0, 0], [1, 0, 0, 0]
        )
        # Off-diagonal
        assert unitary[2, 1] == backend.probability_amplitude(
            unitary, [0, 1, 0, 0], [0, 0, 1, 0]
        )

    def test_probability(self):
        """
        Confirms that expected probability is returned for a single photon
        input.
        """
        backend = PermanentBackend()
        unitary = random_unitary(4)
        # Diagonal
        assert abs(unitary[0, 0]) ** 2 == backend.probability(
            unitary, [1, 0, 0, 0], [1, 0, 0, 0]
        )
        # Off-diagonal
        assert abs(unitary[2, 1]) ** 2 == backend.probability(
            unitary, [0, 1, 0, 0], [0, 0, 1, 0]
        )

    def test_probability_amplitude_multi(self):
        """
        Confirms that expected elements from the unitary matrix are returned
        for a multi photon input.
        """
        backend = PermanentBackend()
        unitary = random_unitary(4, seed=10)
        # Diagonal
        pa = backend.probability_amplitude(unitary, [1, 0, 0, 0], [1, 0, 0, 0])
        assert pa == pytest.approx(0.429095917729817 - 0.366263376556379j, 1e-8)
        # Off-diagonal
        pa = backend.probability_amplitude(unitary, [0, 1, 0, 0], [0, 0, 1, 0])
        assert pa == pytest.approx(
            -0.15003076436547 + 0.4696358907386921j, 1e-8
        )

    def test_probability_multi(self):
        """
        Confirms that expected probability is returned for a multi photon
        input.
        """
        backend = PermanentBackend()
        unitary = random_unitary(4, seed=11)
        # Diagonal
        p = backend.probability(unitary, [1, 0, 0, 0], [1, 0, 0, 0])
        assert p == pytest.approx(0.6122546643219795, 1e-8)
        # Off-diagonal
        p = backend.probability(unitary, [0, 1, 0, 0], [0, 0, 1, 0])
        assert p == pytest.approx(0.25051188442720407, 1e-8)


class TestSlos:
    """
    Specific tests for SLOS function to check for errors in calculation.
    """

    def test_hom(self):
        """Check hom result and ensure returned value is as expected."""
        unitary = np.array([[1, 1j], [1j, 1]]) * 1 / (2**0.5)
        r = SLOSBackend().calculate(unitary, State([1, 1]))
        assert r[1, 1] == 0
        assert r[2, 0] == pytest.approx(0.7071067811865475j)

    def test_known_result(self):
        """
        Check produced output matches a previously calculated result.
        """
        unitary = random_unitary(6, seed=23)
        r = SLOSBackend().calculate(unitary, State([1, 0, 1, 0, 1, 0]))
        assert r[1, 1, 1, 0, 0, 0] == pytest.approx(
            -0.0825807219472892 + 0.0727188703263498j
        )


class TestFockBackend:
    """
    Unit tests for FockBackend baseclass.
    """

    @pytest.mark.parametrize("backend", [PermanentBackend(), SLOSBackend()])
    def test_sampler_results_cached(self, backend):
        """
        Confirms that cached results are used when the same Sampler task is run
        on the backend.
        """
        circuit = Unitary(random_unitary(4))
        # Run initial experiment
        sampler = Sampler(circuit, State([1, 0, 1, 0]), 1000)
        backend.run(sampler)
        # Get data and check if it matches the cache
        results = backend._check_cache(sampler._generate_task())
        assert results is not None
