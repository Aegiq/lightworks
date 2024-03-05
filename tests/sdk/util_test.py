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

from lightworks import random_unitary, random_permutation, Circuit
from lightworks import db_loss_to_transmission, transmission_to_db_loss
from lightworks.sdk import check_unitary
from lightworks.sdk.utils import permutation_to_mode_swaps
from lightworks.sdk.utils import permutation_mat_from_swaps_dict
from lightworks.emulator import set_statistic_type, get_statistic_type

import unittest
from numpy import identity, round
from random import random, seed

class UtilTest(unittest.TestCase):
    """
    Unit tests to check functionality of various utilities included with the
    SDK.
    """
    
    def test_random_unitary(self):
        """
        Checks that when given a seed the random_unitary function always
        produces the same result. If this is not the case it would break many
        of the other unit tests.
        """
        U = random_unitary(4, seed = 111)
        # Check one diagonal element and two off-diagonals
        self.assertAlmostEqual(U[0,0], -0.49007982458868+0.212658840316704j, 8)
        self.assertAlmostEqual(U[1,2], -0.3483593186025-0.683182137239902j, 8)
        self.assertAlmostEqual(U[3,2], 0.12574265147702-0.1257183128681681j, 8)
    
    def test_random_permutation(self):
        """
        Checks that random permutation consistently returns the same results.
        """
        U = random_permutation(4, seed = 222)
        # Check one diagonal element and two off-diagonals
        self.assertEqual(U[0,0], 1+0j)
        self.assertEqual(U[1,2], 0j)
        self.assertEqual(U[3,2], 1+0j)
        
    def test_check_unitary(self):
        """Confirm that the check unitary function behaves as expected."""
        # Check both random unitary and identity matrix
        self.assertTrue(check_unitary(random_unitary(8)))
        self.assertTrue(check_unitary(identity(8)))
        self.assertTrue(check_unitary(identity(8, dtype=complex)))
        
    def test_check_default_statistic(self):
        """Check that the default statistic type is bosonic."""
        self.assertEqual("bosonic", get_statistic_type())
        
    def test_update_statistic(self):
        """
        Check that the statistic type can be switched between bosonic and
        fermionic.
        """
        set_statistic_type("fermionic")
        self.assertEqual("fermionic", get_statistic_type())
        
    def test_incorrect_statistic_update(self):
        """
        Confirms that only valid values can be assigned to statistic type.
        """
        with self.assertRaises(ValueError):
            set_statistic_type("not_bosonic")
            
    def test_permuation_to_swaps(self):
        """Confirms permutation to swaps function produces correct result."""
        perm = random_permutation(4)
        # Determine swaps and create circuit
        swaps = permutation_to_mode_swaps(perm)
        circ = Circuit(4)
        for s in swaps:
            circ.add_bs(s, reflectivity = 0)
        # Check matrix equivalence
        U1 = round(abs(perm)**2, 8)
        U2 = round(abs(circ.U_full)**2, 8)
        self.assertTrue((U1 == U2).all())
        
    def test_swaps_to_permutations(self):
        """
        Checks that conversion from swaps dict to permutation matrix works as 
        expected.
        """
        swaps = {0:2, 2:3, 3:1, 1:0}
        U = permutation_mat_from_swaps_dict(swaps, 4)
        self.assertEqual(abs(U[2,0])**2, 1)
        self.assertEqual(abs(U[3,1])**2, 0)
        self.assertEqual(abs(U[3,2])**2, 1)
        
    def test_db_loss_to_decimal_conv(self):
        """Test conversion from db loss to a decimal transmission value."""
        r = db_loss_to_transmission(0.5)
        self.assertAlmostEqual(r, 0.8912509381337456, 8)
        
    def test_decimal_to_db_loss_conv(self):
        """
        Tests conversion between a decimal transmission and db loss value.
        """
        r = transmission_to_db_loss(0.75)
        self.assertAlmostEqual(r, 1.2493873660829995, 8)
        
    def test_seeded_random(self):
        """
        Checks that the result from the python random module remains consistent
        when using the same seed. If this changes then it could result in other
        unit tests failing."""
        seed(999)
        self.assertAlmostEqual(random(), 0.7813468849570298, 8)
        
    def tearDown(self):
        """Reset stats to bosonic after all tests."""
        set_statistic_type("bosonic")
        return super().tearDown()
        
if __name__ == "__main__":
    unittest.main()