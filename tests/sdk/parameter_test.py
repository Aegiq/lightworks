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

from math import inf

import pytest

from lightworks import (
    Parameter,
    ParameterBoundsError,
    ParameterDict,
    ParameterDictError,
    ParameterValueError,
)


class TestParameter:
    """Unit tests to test functionality of Parameter objects."""

    def test_parameter_creation(self):
        """
        Checks that a Parameter can be created as it's value can retrieved as
        expected.
        """
        p = Parameter(1)
        assert p.get() == 1

    def test_parameter_defaults(self):
        """
        Checks the default values for the different Parameter values when not
        specified.
        """
        p = Parameter(1)
        assert p.label is None
        assert p.min_bound is None
        assert p.max_bound is None

    def test_parameter_update(self):
        """Checks that a Parameter can be updated with set method."""
        p = Parameter(1)
        p.set(2)
        assert p.get() == 2

    def test_bounded_parameter_creation(self):
        """
        Confirms that a Parameter can be created with bounds and also checks an
        error is raised when set bounds don't contain Parameter value.
        """
        p = Parameter(1, bounds=[0, 2])
        assert p.min_bound == 0
        assert p.max_bound == 2
        with pytest.raises(ParameterBoundsError):
            p = Parameter(3, bounds=[0, 2])

    def test_bounded_parameter_update(self):
        """
        Checks that a bounded Parameter can be updated with set method, and
        also confirms an error is raised when parameters are set outside the
        bounds.
        """
        p = Parameter(1, bounds=[0, 3])
        p.set(2)
        assert p.get() == 2
        with pytest.raises(ParameterValueError):
            p.set(-1)
        with pytest.raises(ParameterValueError):
            p.set(4)

    def test_invalid_bound_update(self):
        """
        Checks that an error is raised when bounds are updated such that the
        current Parameter value would become invalid.
        """
        p = Parameter(2, bounds=[0, 3])
        with pytest.raises(ParameterBoundsError):
            p.max_bound = 1
        with pytest.raises(ParameterBoundsError):
            p.min_bound = 3

    def test_non_numerical_value(self):
        """
        Checks non-numerical values can be assigned for Parameter objects and
        also checks an error is raised if bounds are to be added.
        """
        p = Parameter(True)
        p.set(False)
        assert not p.get()
        with pytest.raises(ParameterBoundsError):
            p = Parameter(True, bounds=[0, 1])

    def test_non_numerical_bound_update(self):
        """
        Confirms that non-numerical Parameter cannot have bounds assigned after
        creation.
        """
        p = Parameter(True)
        with pytest.raises(ParameterBoundsError):
            p.min_bound = 0

    def test_bounded_value_type_updated(self):
        """
        Checks that a numerical Parameter with bounds cannot be switched to
        another non-numerical value.
        """
        p = Parameter(1, bounds=[0, 2])
        with pytest.raises(ParameterValueError):
            p.set("Test")

    def test_has_bounds_method(self):
        """Checks behaviour of the has_bounds method is correct."""
        p = Parameter(1)
        assert not p.has_bounds()
        p = Parameter(1, bounds=[0, 2])
        assert p.has_bounds()

    def test_str_return(self):
        """
        Confirms that Parameter value is returned when str function is used on
        Parameter.
        """
        p = Parameter(1.2)
        assert str(p) == "1.2"

    @pytest.mark.parametrize("value", [1, 2.1, "test"])
    def test_repr_return(self, value):
        """
        Checks Parameter value is contained within the value returned when repr
        is used.
        """
        p = Parameter(value)
        assert str(value) in repr(p)

    def test_label_in_repr_return(self):
        """Checks label included in repr return."""
        p = Parameter(1, label="label")
        assert "label" in repr(p)

    def test_bounds_in_repr_return(self):
        """Checks bounds in repr return."""
        p = Parameter(1, bounds=[0, 3])
        assert str(p.min_bound) in repr(p)
        assert str(p.max_bound) in repr(p)


class TestParameterDict:
    """Unit tests to test functionality of ParameterDict objects."""

    def test_dict_creation(self):
        """Checks dictionary can be created and is empty after creation."""
        pd = ParameterDict()
        assert pd.params == []

    def test_dict_assignment(self):
        """Check initial assignment of Parameter into dictionary."""
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        assert pd["a"].get() == 1

    def test_dict_return(self):
        """Confirms that the return when using [] is a Parameter object."""
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        assert isinstance(pd["a"], Parameter)

    def test_dict_invalid_param(self):
        """
        Confirms that a KeyError is raised when a parameter key does not exist.
        """
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        with pytest.raises(KeyError):
            pd["b"]

    def test_parameter_update(self):
        """Test updating of parameter through assigning new value to dict."""
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        pd["a"] = 2
        assert pd["a"].get() == 2

    def test_incorrect_set_item(self):
        """
        Checks an error is raised when a non-Parameter object is assigned to a
        new key and when it is attempted to overwrite a Parameter object in the
        dictionary with another Parameter.
        """
        pd = ParameterDict()
        with pytest.raises(ParameterDictError):
            pd["a"] = 1
        pd["a"] = Parameter(1)
        with pytest.raises(ParameterDictError):
            pd["a"] = Parameter(2)

    def test_len_method(self):
        """Confirms len method produces the correct value."""
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        pd["b"] = Parameter(2)
        pd["c"] = Parameter(3)
        assert len(pd) == 3

    def test_has_bounds_method(self):
        """
        Checks has_bounds method returns correct values when bounds are/not
        present.
        """
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        pd["b"] = Parameter(2)
        assert not pd.has_bounds()
        pd["a"].min_bound = 0
        assert pd.has_bounds()

    def test_params_property(self):
        """
        Confirms correct values are returned by params attribute of class.
        """
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        pd["b"] = Parameter(2)
        assert pd.params == ["a", "b"]

    def test_params_modification(self):
        """Checks that params attribute cannot be modified."""
        pd = ParameterDict()
        with pytest.raises(AttributeError):
            pd.params = ["c"]

    def test_in_operator(self):
        """Tests that key in ParameterDict returns the correct values."""
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        assert "a" in pd
        assert "b" not in pd
        assert "b" not in pd

    def test_remove_method(self):
        """
        Checks that remove method of ParameterDict is able to remove a
        parameter from the dictionary.
        """
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        pd["b"] = Parameter(2)
        pd.remove("a")
        assert "a" not in pd

    def test_iterable_behavior(self):
        """
        Confirms that the values when using the class as an iterable is the
        parameter keys.
        """
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        pd["b"] = Parameter(2)
        pd["c"] = Parameter(2)
        # Loop over dictionary and store returned values
        params = []
        for p in pd:
            params.append(p)
        # Check lists is equivalent to params attribute
        assert params == pd.params

    def test_get_bounds(self):
        """
        Confirms that get bounds is able to identify bounds for parameter.
        """
        pd = ParameterDict()
        pd["a"] = Parameter(1, bounds=[0, 2])
        assert pd.get_bounds()["a"] == (0, 2)

    def test_get_bounds_no_bounds(self):
        """
        Confirms that get bounds replaces None bounds with +/- inf.
        """
        pd = ParameterDict()
        pd["a"] = Parameter(1)
        assert pd.get_bounds()["a"] == (-inf, inf)
