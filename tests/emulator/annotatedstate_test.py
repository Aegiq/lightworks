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
from lightworks.emulator import AnnotatedStateError
from lightworks.emulator.state import AnnotatedState
from lightworks.emulator.utils.state import annotated_state_to_string


class TestAnnotatedState:
    """Unit tests for AnnotatedState object."""

    def test_get_mode_value(self):
        """Confirms that mode value retrieval works as expected."""
        s = AnnotatedState([[], [0], [1], []])
        assert s[1] == [0]

    def test_state_equality(self):
        """Checks that state equality comparison works as expected."""
        assert AnnotatedState([[], [0], [1], []]) == AnnotatedState(
            [[], [0], [1], []]
        )
        assert AnnotatedState([[], [0], [1], []]) != AnnotatedState(
            [[], [1], [0], []]
        )

    def test_state_equality_diff_types(self):
        """
        Checks that state equality returns False when compared with a
        non-AnnotatedState object.
        """
        s = AnnotatedState([[0], [1], [2]])
        assert s != [[0], [1], [2]]

    def test_state_equality_shuffled(self):
        """
        Checks that state equality comparison works as expected in the case
        that identical labels are provide in a mode but in a different order.
        """
        assert AnnotatedState([[], [0], [1, 2], []]) == AnnotatedState(
            [[], [0], [2, 1], []]
        )

    def test_state_addition(self):
        """Checks that state addition behaviour is as expected."""
        s = AnnotatedState([[0], [2, 3], [1]]) + AnnotatedState(
            [[], [0, 3], [1]]
        )
        assert s == AnnotatedState([[0], [2, 3], [1], [], [0, 3], [1]])
        s = AnnotatedState([[0], [2, 3], [1]]) + AnnotatedState([])
        assert s == AnnotatedState([[0], [2, 3], [1]])

    def test_merge(self):
        """Checks that state merge works correctly."""
        s = AnnotatedState([[0], [2, 3], [1]]).merge(
            AnnotatedState([[], [0], [1]])
        )
        assert s == AnnotatedState([[0], [2, 3, 0], [1, 1]])

    def test_modification_behavior(self):
        """
        Checks that the correct error is raised if we try to modify the State
        value. Also tests what happens when state s attribute is modified.
        """
        s = AnnotatedState([[0, 1], [2]])
        with pytest.raises(AnnotatedStateError):
            s.s = [[0]]
        s.s[0] = [2]
        assert s == AnnotatedState([[0, 1], [2]])

    def test_mode_length(self):
        """Check the calculated mode number attribute is set correctly."""
        assert len(AnnotatedState([[], [0], [1, 2], [], [0]])) == 5
        assert len(AnnotatedState([[], [], [], []])) == 4
        assert len(AnnotatedState([])) == 0

    def test_photon_number(self):
        """Checks calculated photon number value is correct."""
        assert AnnotatedState([[], [0], [1, 2], [], [0]]).n_photons == 4
        assert AnnotatedState([[], [], [], []]).n_photons == 0
        assert AnnotatedState([]).n_photons == 0

    @pytest.mark.parametrize(
        "value", [[0, 1, 2], ["0", "1", "2"], [[0], 1], [0], [None]]
    )
    def test_req_list_of_lists(self, value):
        """
        Checks that a exception is raised in Annotated state isn't provided a
        list of lists as the input.
        """
        with pytest.raises(TypeError):
            AnnotatedState(value)

    def test_n_modes_modification(self):
        """
        Tests that an exception is raised when a user attempts to modify an
        AnnotatedState.
        """
        s = AnnotatedState([[0], [1], [2]])
        with pytest.raises(AnnotatedStateError):
            s.n_modes = 2

    def test_merge_not_same_length(self):
        """
        Confirms that merge method raises an exception when AnnotatedStates are
        not of the same length.
        """
        s = AnnotatedState([[0], [1], [2]])
        s2 = AnnotatedState([[0], [1], [2], [3]])
        with pytest.raises(ValueError):
            s.merge(s2)
        with pytest.raises(ValueError):
            s2.merge(s)

    @pytest.mark.parametrize("value", [(0, 1, 2), [0, 1, 2], State([0, 1, 2])])
    def test_add_not_same_type(self, value):
        """
        Checks that an exception is raised when a non-AnnotatedState is
        attempted to be added to an AnnotatedState
        """
        s = AnnotatedState([[0], [1], [2]])
        with pytest.raises(TypeError):
            s + value

    def test_str_returns_correct_value(self):
        """
        Checks that string function returns correct value for AnnotatedState.
        """
        s = [[0], [1], [2]]
        assert str(AnnotatedState(s)) == annotated_state_to_string(s)

    def test_repr_returns_correct_value(self):
        """
        Checks that repr function contains correct value for AnnotatedState.
        """
        s = [[0], [1], [2]]
        assert annotated_state_to_string(s) in str(AnnotatedState(s))

    def test_set_item_blocked(self):
        """
        Confirms that set item functionality is blocked for the AnnotatedState.
        """
        s = AnnotatedState([[0], [1], [2]])
        with pytest.raises(AnnotatedStateError):
            s[0] = [1]

    def test_get_item_slicing(self):
        """
        Checks that slicing can be used to get a part of an AnnotatedState.
        """
        s = AnnotatedState([[0], [1], [2], [3]])
        assert s[0:2] == AnnotatedState([[0], [1]])

    @pytest.mark.parametrize("value", [[0, 1], None, "1"])
    def test_get_item_invalid_value(self, value):
        """
        Confirms error raised when an invalid value is provided to get item.
        """
        s = AnnotatedState([[0], [1], [2], [3]])
        with pytest.raises(TypeError):
            s[value]
