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

from random import random

import numpy as np
import pytest

from lightworks import Circuit, Unitary, random_unitary
from lightworks.interferometers import ErrorModel, Reck
from lightworks.interferometers.decomposition import (
    bs_matrix,
    check_null,
    reck_decomposition,
)


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

    @pytest.mark.parametrize("value", ["not_error_model", Circuit(4), 0])
    def test_error_model_invalid_type(self, value):
        """
        Checks that an exception is raised if the error_model is set to
        something other than an ErrorModel or None.
        """
        with pytest.raises(TypeError):
            Reck(error_model=value)


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
            assert em.get_bs_reflectivity() == 0.5

    def test_default_loss(self):
        """
        Checks that default loss value is 0.
        """
        em = ErrorModel()
        # Repeat 100 times to confirm no randomness present
        for _i in range(100):
            assert em.get_loss() == 0


class TestDecomposition:
    """
    Tests for decomposition module.
    """

    @pytest.mark.parametrize("n_modes", [2, 7, 8])
    def test_decomposition(self, n_modes):
        """
        Checks decomposition is able to pass successfully for a valid unitary
        matrix.
        """
        unitary = random_unitary(n_modes)
        reck_decomposition(unitary)

    @pytest.mark.parametrize("n_modes", [2, 7, 8])
    def test_decomposition_identity(self, n_modes):
        """
        Checks decomposition is able to pass successfully for an identity
        matrix.
        """
        unitary = np.identity(n_modes, dtype=complex)
        reck_decomposition(unitary)

    @pytest.mark.parametrize("n_modes", [2, 7, 8])
    def test_decomposition_failed(self, n_modes):
        """
        Checks decomposition fails for a non-unitary matrix.
        """
        unitary = np.zeros((n_modes, n_modes), dtype=complex)
        for i in range(n_modes):
            for j in range(n_modes):
                unitary[i, j] = random() + 1j * random()
        with pytest.raises(ValueError):
            reck_decomposition(unitary)

    def test_bs_matrix(self):
        """
        Check beam splitter matrix is correct for the unit cell used.
        """
        theta, phi = 2 * np.pi * random(), 2 * np.pi * random()
        # Get beam splitter matrix
        bs_u = bs_matrix(0, 1, theta, phi, 2)
        # Create unit cell circuit
        circ = Circuit(2)
        circ.add_ps(0, phi)
        circ.add_bs(0)
        circ.add_ps(1, theta)
        circ.add_bs(0)
        circ_u = circ.U
        # Check equivalence
        assert (bs_u.round(8) == circ_u.round(8)).all()

    @pytest.mark.parametrize("n_modes", [2, 7, 8])
    def test_check_null(self, n_modes):
        """
        Checks null matrix returns True for a diagonal matrix.
        """
        unitary = np.identity(n_modes, dtype=complex)
        for i in range(n_modes):
            unitary[i, i] *= np.exp(1j * random())
        assert check_null(unitary)

    def test_check_null_false(self):
        """
        Checks null matrix returns false for a non-nulled matrix.
        """
        unitary = random_unitary(8)
        assert not check_null(unitary)
