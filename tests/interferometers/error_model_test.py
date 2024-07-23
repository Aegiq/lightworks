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

from lightworks.interferometers import ErrorModel
from lightworks.interferometers.dists import (
    Distribution,
    Gaussian,
    TopHat,
)


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

    @pytest.mark.parametrize("dist", [Gaussian(0.5, 0.1), TopHat(0.1, 0.9)])
    def test_set_random_seed(self, dist):
        """
        Checks that set random seed produces identical values on subsequent
        calls when using each Distribution object with an rng.
        """
        em = ErrorModel()
        em.bs_reflectivity = dist
        # Set seed and get values twice
        em._set_random_seed(11)
        v1 = em.get_bs_reflectivity()
        em._set_random_seed(11)
        v2 = em.get_bs_reflectivity()
        # Check equivalence
        assert v1 == v2

    def test_set_random_seed_identical_distributions(self):
        """
        Checks that set random seed produces different values when two
        quantities are assigned to identical distributions.
        """
        em = ErrorModel()
        em.bs_reflectivity = Gaussian(0.5, 0.1)
        em.loss = Gaussian(0.5, 0.1)
        # Set seed and get values reflectivity and loss
        em._set_random_seed(11)
        v1 = em.get_bs_reflectivity()
        em._set_random_seed(11)
        v2 = em.get_loss()
        # Check equivalence
        assert v1 != v2

    def test_bs_reflectivity_is_distribution(self):
        """
        Checks that return by bs_reflectivity property is a Distribution object.
        """
        assert isinstance(ErrorModel().bs_reflectivity, Distribution)

    @pytest.mark.parametrize("value", [1, True, None, "Distribution"])
    def test_bs_reflectivity_enforces_distribution(self, value):
        """
        Checks that bs_reflectivity setting enforces that the value is a
        Distribution object.
        """
        em = ErrorModel()
        with pytest.raises(TypeError):
            em.bs_reflectivity = value

    def test_loss_is_distribution(self):
        """
        Checks that return by loss property is a Distribution object.
        """
        assert isinstance(ErrorModel().loss, Distribution)

    @pytest.mark.parametrize("value", [1, True, None, "Distribution"])
    def test_loss_enforces_distribution(self, value):
        """
        Checks that setting loss setting enforces that the value is a
        Distribution object.
        """
        em = ErrorModel()
        with pytest.raises(TypeError):
            em.loss = value

    def test_phase_offset_is_distribution(self):
        """
        Checks that return by phase_offset property is a Distribution object.
        """
        assert isinstance(ErrorModel().phase_offset, Distribution)

    @pytest.mark.parametrize("value", [1, True, None, "Distribution"])
    def test_phase_offset_enforces_distribution(self, value):
        """
        Checks that setting phase_offset setting enforces that the value is a
        Distribution object.
        """
        em = ErrorModel()
        with pytest.raises(TypeError):
            em.phase_offset = value

    @pytest.mark.parametrize("param", ["bs_reflectivity", "loss"])
    def test_parameters_in_string_and_repr(self, param):
        """
        Checks that all parameters of the error model are contained in the
        string and repr returns.
        """
        em = ErrorModel()
        assert param in str(em)
        assert param in repr(em)

    @pytest.mark.parametrize(
        "value", ["bs_reflectivity", "loss", "phase_offset"]
    )
    def test_random_seeding(self, value):
        """
        Checks that set random seed produces different values when two
        quantities are assigned to identical distributions.
        """
        em = ErrorModel()
        setattr(em, value, Gaussian(0.5, 0.1))
        # Set seed and get value twice
        em._set_random_seed(11)
        v1 = getattr(em, value).value()
        em._set_random_seed(11)
        v2 = getattr(em, value).value()
        # Check equivalence
        assert v1 == v2
