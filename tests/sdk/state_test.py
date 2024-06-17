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

from lightworks import State
from lightworks.sdk.utils import StateError


class TestState:
    """Unit tests for State object."""

    def test_get_mode_value(self):
        """Confirms that mode value retrieval works as expected."""
        s = State([1, 2, 3])
        assert s[1] == 2

    def test_state_equality(self):
        """Checks that state equality comparison works as expected."""
        assert State([1, 0, 1, 0]) == State([1, 0, 1, 0])
        assert State([1, 0, 1, 0]) != State([0, 1, 1, 0])

    def test_state_addition(self):
        """Checks that state addition behaviour is as expected."""
        s = State([1, 2, 3]) + State([0, 4, 5])
        assert s == State([1, 2, 3, 0, 4, 5])
        s = State([1, 2, 3]) + State([])
        assert s == State([1, 2, 3])

    def test_merge(self):
        """Checks that state merge works correctly."""
        s = State([1, 2, 3]).merge(State([0, 2, 1]))
        assert s == State([1, 4, 4])

    def test_modification_behavior(self):
        """
        Checks that the correct error is raised if we try to modify the State
        value and checks behaviour if the s attribute of the state is modified.
        """
        s = State([1, 2, 3])
        with pytest.raises(StateError):
            s.s = [1]
        s.s[0] = 2
        assert s == State([1, 2, 3])

    def test_mode_length(self):
        """Check the calculated mode number attribute is set correctly."""
        assert len(State([1, 2, 0, 1, 0])) == 5
        assert len(State([0, 0, 0, 0])) == 4
        assert len(State([])) == 0

    def test_photon_number(self):
        """Checks calculated photon number value is correct."""
        assert State([1, 0, 2, 1, 1]).n_photons == 5
        assert State([0, 0, 0, 0, 0]).n_photons == 0
        assert State([]).n_photons == 0

    @pytest.mark.parametrize(
        "state", [[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 2, 0], [0, 0, 0, 0]]
    )
    def test_pass_validation(self, state):
        """
        Checks that the _validate method of state does not raise an exception
        when an valid configuration is provided.
        """
        s = State(state)
        s._validate()

    @pytest.mark.parametrize(
        "state", [[1, 0, -1, 0], [1, 1, -1, 0], [1, 0, 0.5, 0], [1.0, 0, 0, 0]]
    )
    def test_fail_validation(self, state):
        """
        Checks that the _validate method of state raises an exception when an
        invalid configuration is provided.
        """
        s = State(state)
        with pytest.raises((ValueError, TypeError)):
            s._validate()
