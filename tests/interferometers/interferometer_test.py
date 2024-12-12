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

from lightworks import PhotonicCircuit, Unitary, random_unitary
from lightworks.interferometers import ErrorModel, Reck
from lightworks.interferometers.dists import Gaussian, TopHat


class TestReck:
    """
    Tests to check functionality of the Reck interferometer.
    """

    @pytest.mark.parametrize("n_modes", [2, 3, 7, 8, 15, 16])
    def test_equivalence(self, n_modes):
        """
        Checks map functionality produces an equivalent circuit for a range of
        mode values.
        """
        # Create test circuit
        test_circ = Unitary(random_unitary(n_modes))
        # Find mapped circuit
        mapped_circ = Reck().map(test_circ)
        # Then check equivalence
        assert (test_circ.U.round(8) == mapped_circ.U.round(8)).all()

    def test_sequential_maps(self):
        """
        Checks random procedure produces different circuits on subsequent calls
        when the random seed is not set.
        """
        # Create test circuit
        test_circ = Unitary(random_unitary(8))
        # Define error model with random variations
        emodel = ErrorModel()
        emodel.bs_reflectivity = Gaussian(0.5, 0.02, min_value=0, max_value=1)
        emodel.loss = TopHat(0.1, 0.2)
        emodel.phase_offset = Gaussian(0, 0.02)
        r = Reck(emodel)
        # Create two mapped circuits
        mapped_circ = r.map(test_circ)
        mapped_circ2 = r.map(test_circ)
        # Then check equivalence
        assert (mapped_circ.U.round(8) != mapped_circ2.U.round(8)).any()

    def test_random_seeding(self):
        """
        Checks random seeding produces repeatable circuits for a range of mode
        values.
        """
        # Create test circuit
        test_circ = Unitary(random_unitary(8))
        # Define error model with random variations
        emodel = ErrorModel()
        emodel.bs_reflectivity = Gaussian(0.5, 0.02, min_value=0, max_value=1)
        emodel.loss = TopHat(0.1, 0.2)
        emodel.phase_offset = Gaussian(0, 0.02)
        r = Reck(emodel)
        # Set seed and create two mapped circuits
        mapped_circ = r.map(test_circ, seed=12)
        mapped_circ2 = r.map(test_circ, seed=12)
        # Then check equivalence
        assert (mapped_circ.U.round(8) == mapped_circ2.U.round(8)).all()

    @pytest.mark.parametrize(
        "value", ["not_error_model", PhotonicCircuit(4), 0]
    )
    def test_error_model_invalid_type(self, value):
        """
        Checks that an exception is raised if the error_model is set to
        something other than an ErrorModel or None.
        """
        with pytest.raises(TypeError):
            Reck(error_model=value)
