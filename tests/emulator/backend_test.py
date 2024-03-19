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

from lightworks import random_unitary, Unitary, State
from lightworks.emulator import Backend

import pytest

class TestBackend:
    """
    Unit tests for ensuring backend object remains functioning correctly.
    """
    
    def test_valid_backends(self):
        """Checks all currently valid backends can be set."""
        
        for backend in ["permanent", "slos"]:
            Backend(backend)
            
    def test_invalid_backends(self):
        """Checks an invalid backend raises a ValueError"""
        
        with pytest.raises(ValueError):
            Backend("not_a_backend")
            
    def test_probability_amplitude(self):
        """
        Confirms that expected elements from the unitary matrix are returned 
        for a single photon input.
        """
        backend = Backend("permanent")
        U = random_unitary(4)
        # Diagonal
        assert U[0,0] == backend.probability_amplitude(U, [1,0,0,0], [1,0,0,0])
        # Off-diagonal
        assert U[2,1] == backend.probability_amplitude(U, [0,1,0,0], [0,0,1,0])
        
    def test_probability(self):
        """
        Confirms that expected probability is returned for a single photon 
        input.
        """
        backend = Backend("permanent")
        U = random_unitary(4)
        # Diagonal
        assert abs(U[0,0])**2 == backend.probability(U, [1,0,0,0], [1,0,0,0])
        # Off-diagonal
        assert abs(U[2,1])**2 == backend.probability(U, [0,1,0,0], [0,0,1,0])
        
    def test_probability_amplitude_multi(self):
        """
        Confirms that expected elements from the unitary matrix are returned 
        for a multi photon input.
        """
        backend = Backend("permanent")
        U = random_unitary(4, seed = 10)
        # Diagonal
        pa = backend.probability_amplitude(U, [1,0,0,0], [1,0,0,0])
        assert pa == pytest.approx(0.429095917729817-0.366263376556379j, 1e-8)
        # Off-diagonal
        pa = backend.probability_amplitude(U, [0,1,0,0], [0,0,1,0])
        assert pa == pytest.approx(-0.15003076436547+0.4696358907386921j, 1e-8)
        
    def test_probability_multi(self):
        """
        Confirms that expected probability is returned for a multi photon 
        input.
        """
        backend = Backend("permanent")
        U = random_unitary(4, seed = 11)
        # Diagonal
        p = backend.probability(U, [1,0,0,0], [1,0,0,0])
        assert p == pytest.approx(0.6122546643219795, 1e-8)
        # Off-diagonal
        p = backend.probability(U, [0,1,0,0], [0,0,1,0])
        assert p == pytest.approx(0.25051188442720407, 1e-8)
        
    def test_full_probability_distribution(self):
        """
        Check against a known result for the full probability distribution
        while using the permanent backend.
        """
        backend = Backend("permanent")
        U = random_unitary(3, seed = 2)
        dist = backend.full_probability_distribution(Unitary(U)._build(), 
                                                     State([1,1,0]))
        known_dist = {State([2,0,0]): 0.2600968016733883,
                      State([1,1,0]): 0.22397644081164791,
                      State([0,2,0]): 0.13677578896115924,
                      State([1,0,1]): 0.04763948802896683,
                      State([0,1,1]): 0.2377381500275615,
                      State([0,0,2]): 0.09377333049727585}
        for d in dist:
            if d not in known_dist:
                pytest.fail(
                    "State from produced distribution not in known "
                    "distribution.")
            assert known_dist[d] == pytest.approx(dist[d], 1e-8)
        
    def test_permanent_slos_equivalence(self):
        """
        Check permanent and slos backends produce equivalent probability
        distributions.
        """
        circuit = Unitary(random_unitary(6))
        input_state = State([1,0,1,0,1,0])
        # Find distributions
        backend_p = Backend("permanent")
        p1 = backend_p.full_probability_distribution(circuit._build(),
                                                     input_state)
        backend_s = Backend("permanent")
        p2 = backend_s.full_probability_distribution(circuit._build(),
                                                     input_state)
        # Check equivalence
        for s in p1:
            if round(p1[s], 8) != round(p2[s], 8):
                pytest.fail("Methods do not produce equivalent distributions.")