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

from lightworks.interferometers.dists import (
    Constant,
    Distribution,
    Gaussian,
    TopHat,
)
from lightworks.interferometers.dists.utils import is_number


class TestDistributions:
    """General tests for probability distribution (dists) module."""

    @pytest.mark.parametrize(
        "dist",
        [Constant(0.5), Gaussian(0.5, 0), TopHat(0.5, 0.5)],
    )
    def test_is_distribution(self, dist):
        """
        Checks each of the distribution classes is an instance of the
        distribution base class.
        """
        assert isinstance(dist, Distribution)

    @pytest.mark.parametrize(
        "value", [1, 0.5, np.inf, np.float64(1.2), [1, 0.5, np.inf]]
    )
    def test_is_number(self, value):
        """
        Confirms that is_number does not return an exception for valid values.
        """
        is_number(value)

    @pytest.mark.parametrize("value", ["1", None, True, [1, True]])
    def test_is_number_invalid(self, value):
        """
        Confirms that is_number returns an exception when an invalid value is
        attempted to be set.
        """
        with pytest.raises(TypeError):
            is_number(value)

    def test_custom_distribution_requires_value(self):
        """
        Checks that any created custom class which uses Distribution as the
        parent requires the value method for the class to be initialized.
        """

        class TestClass(Distribution):
            pass

        with pytest.raises(TypeError):
            TestClass()


class TestConstant:
    """Tests for Constant distribution"""

    def test_constant(self):
        """
        Checks that Constant distribution just returns set value.
        """
        val = random()
        c = Constant(val)
        # Check 100 times to ensure it always works
        for _i in range(100):
            assert val == c.value()

    def test_params_in_string_and_repr(self):
        """
        Checks that the assigned value is detailed in the string and repr.
        """
        val = random()
        c = Constant(val)
        assert str(val) in str(c)
        assert str(val) in repr(c)


class TestGaussian:
    """Tests for Constant distribution"""

    @pytest.mark.flaky(max_runs=3)
    def test_gaussian(self):
        """
        Checks that Gaussian distribution generates values with a mean close to
        the set value for large numbers.
        """
        dist = Gaussian(1, 0.2)
        vals = [dist.value() for _i in range(100000)]
        # Check within 5% of expected mean
        assert np.mean(vals) == pytest.approx(1, 0.05)

    def test_max_less_than_min(self):
        """
        Confirms that exception is raised if the max value is set lower than
        the min value
        """
        with pytest.raises(ValueError):
            Gaussian(1, 0.2, min_value=1, max_value=0)

    def test_values_within_set_bounds(self):
        """
        Checks that values are forced to exist within the set bounds by setting
        bounds to exist within a narrow window of the Gaussian distribution.
        """
        dist = Gaussian(1, 0.3, min_value=1, max_value=1.5)
        vals = [dist.value() for _i in range(100000)]
        # Check values are within the minimum and maximum range
        assert min(vals) >= 1
        assert max(vals) <= 1.5

    def test_random_seed(self):
        """
        Checks that random seed produce repeatable values.
        """
        dist = Gaussian(1, 0.3)
        # Set seed and sampler twice
        dist.set_random_seed(99)
        v1 = dist.value()
        dist.set_random_seed(99)
        v2 = dist.value()
        # Check equivalence
        assert v1 == v2

    def test_params_in_string_and_repr(self):
        """
        Checks that the assigned value is detailed in the string and repr.
        """
        val = random()
        dist = Gaussian(val, 2 * val)
        assert str(val) in str(dist)
        assert str(2 * val) in str(dist)
        assert str(val) in repr(dist)
        assert str(2 * val) in repr(dist)

    def test_params_in_string_and_repr_bounds(self):
        """
        Checks that the assigned min and max value is detailed in the string and
        repr.
        """
        val = random()
        dist = Gaussian(1, 0.5, min_value=val, max_value=2 * val)
        assert str(val) in str(dist)
        assert str(2 * val) in str(dist)
        assert str(val) in repr(dist)
        assert str(2 * val) in repr(dist)


class TestTopHat:
    """Tests for TopHat distribution"""

    def test_top_hat(self):
        """
        Checks that Top Hat distribution only creates values within the expected
        range.
        """
        dist = TopHat(0.4, 0.6)
        vals = [dist.value() for _i in range(100000)]
        # Check min and max value within set range
        assert min(vals) >= 0.4
        assert max(vals) <= 0.6

    def test_max_less_than_min(self):
        """
        Confirms that exception is raised if the max value is set lower than
        the min value
        """
        with pytest.raises(ValueError):
            TopHat(0.6, 0.4)

    def test_random_seed(self):
        """
        Checks that random seed produce repeatable values.
        """
        dist = TopHat(0.4, 0.6)
        # Set seed and sampler twice
        dist.set_random_seed(99)
        v1 = dist.value()
        dist.set_random_seed(99)
        v2 = dist.value()
        # Check equivalence
        assert v1 == v2

    def test_params_in_string_and_repr(self):
        """
        Checks that the assigned value is detailed in the string and repr.
        """
        val = random()
        dist = TopHat(val, 2 * val)
        assert str(val) in str(dist)
        assert str(2 * val) in str(dist)
        assert str(val) in repr(dist)
        assert str(2 * val) in repr(dist)
