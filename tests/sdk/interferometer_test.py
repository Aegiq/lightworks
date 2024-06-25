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

from lightworks import Unitary, random_unitary
from lightworks.sdk.interferometers import ErrorModel, Reck


class TestReck:
    """
    Unit tests to check functionality of the Reck interferometer.
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


class TestErrorModel:
    """
    Tests for Error Model object of module.
    """

    def test_default_bs_reflectivity(self):
        """
        Checks that the default beam splitter reflectivity is 0.5.
        """
        em = ErrorModel()
        # Repeat 100 times to confirm no randomness present
        for _i in range(100):
            assert em.bs_reflectivity == 0.5

    def test_default_loss(self):
        """
        Checks that default loss value is 0.
        """
        em = ErrorModel()
        # Repeat 100 times to confirm no randomness present
        for _i in range(100):
            assert em.loss == 0
