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

from random import randint, random, seed

import numpy as np
import pytest
import sympy as sp

from lightworks import (
    Parameter,
    PostSelection,
    PostSelectionFunction,
    State,
    convert,
    random_permutation,
    random_unitary,
    settings,
)
from lightworks.sdk.circuit.photonic_circuit_utils import (
    add_mode_to_unitary,
    check_loss,
)
from lightworks.sdk.utils.heralding import add_heralds_to_state
from lightworks.sdk.utils.matrix import check_unitary
from lightworks.sdk.utils.param_unitary import ParameterizedUnitary
from lightworks.sdk.utils.permutation import permutation_mat_from_swaps_dict
from lightworks.sdk.utils.post_selection import (
    DefaultPostSelection,
    PostSelectionType,
    Rule,
)
from lightworks.sdk.utils.random import process_random_seed


class TestUtils:
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
        unitary = random_unitary(4, seed=111)
        # Check one diagonal element and two off-diagonals
        assert unitary[0, 0] == pytest.approx(
            -0.49007982458868 + 0.212658840316704j, 1e-8
        )
        assert unitary[1, 2] == pytest.approx(
            -0.3483593186025 - 0.683182137239902j, 1e-8
        )
        assert unitary[3, 2] == pytest.approx(
            0.12574265147702 - 0.1257183128681681j, 1e-8
        )

    def test_random_permutation(self):
        """
        Checks that random permutation consistently returns the same results.
        """
        unitary = random_permutation(4, seed=44)
        # Check one diagonal element and two off-diagonals
        assert unitary[0, 0] == 0j
        assert unitary[0, 1] == 1 + 0j
        assert unitary[3, 0] == 0j

    def test_check_unitary(self):
        """Confirm that the check unitary function behaves as expected."""
        # Check both random unitary and identity matrix
        assert check_unitary(random_unitary(8))
        assert check_unitary(np.identity(8))
        assert check_unitary(np.identity(8, dtype=complex))

    def test_swaps_to_permutations(self):
        """
        Checks that conversion from swaps dict to permutation matrix works as
        expected.
        """
        swaps = {0: 2, 2: 3, 3: 1, 1: 0}
        unitary = permutation_mat_from_swaps_dict(swaps, 4)
        assert abs(unitary[2, 0]) ** 2 == 1
        assert abs(unitary[3, 1]) ** 2 == 0
        assert abs(unitary[3, 2]) ** 2 == 1

    def test_db_loss_to_decimal_conv(self):
        """Test conversion from db loss to a decimal loss value."""
        r = 1 - convert.db_loss_to_decimal(0.5)
        assert r == pytest.approx(0.8912509381337456, 1e-8)

    def test_decimal_to_db_loss_conv(self):
        """
        Tests conversion between a decimal loss and db loss value.
        """
        r = convert.decimal_to_db_loss(0.25)
        assert r == pytest.approx(1.2493873660829995, 1e-8)

    def test_seeded_random(self):
        """
        Checks that the result from the python random module remains consistent
        when using the same seed. If this changes then it could result in other
        unit tests failing.
        """
        seed(999)
        assert random() == pytest.approx(0.7813468849570298, 1e-8)

    @pytest.mark.parametrize("mode", [0, 3, 5, 6])
    def test_add_mode_to_unitary_value(self, mode):
        """
        Checks that add_mode_to_unitary function works correctly for a variety
        of positions, producing a value of 1 in the expected position.
        """
        unitary = random_unitary(6)
        new_unitary = add_mode_to_unitary(unitary, mode)
        # Check diagonal value on new mode
        assert new_unitary[mode, mode] == 1.0
        # Also confirm one off-diagonal value
        assert new_unitary[mode, mode - 1] == 0.0

    @pytest.mark.parametrize("mode", [0, 3, 5, 6])
    def test_add_mode_to_unitary_diagonal(self, mode):
        """
        Checks that add_mode_to_unitary function works correctly for a variety
        of positions.
        """
        unitary = random_unitary(6)
        new_unitary = add_mode_to_unitary(unitary, mode)
        # Confirm preservation of unitary on diagonal
        assert (new_unitary[:mode, :mode] == unitary[:mode, :mode]).all()
        assert (
            new_unitary[mode + 1 :, mode + 1 :] == unitary[mode:, mode:]
        ).all()

    @pytest.mark.parametrize("mode", [0, 3, 5, 6])
    def test_add_mode_to_unitary_off_diagonal(self, mode):
        """
        Checks that add_mode_to_unitary function works correctly for a variety
        of positions.
        """
        unitary = random_unitary(6)
        new_unitary = add_mode_to_unitary(unitary, mode)
        # Confirm preservation on unitary off diagonal
        assert (new_unitary[mode + 1 :, :mode] == unitary[mode:, :mode]).all()
        assert (new_unitary[:mode, mode + 1 :] == unitary[:mode, mode:]).all()

    def test_add_mode_to_parameterized_unitary_return(self):
        """
        Checks that add_mode_to_unitary function returns a Parameterized unitary
        when one is supplied.
        """
        unitary = ParameterizedUnitary(
            sp.Matrix([[6 * i + j for j in range(6)] for i in range(6)]), {}
        )
        new_unitary = add_mode_to_unitary(unitary, 2)
        assert isinstance(new_unitary, ParameterizedUnitary)

    @pytest.mark.parametrize("mode", [0, 3, 5, 6])
    def test_add_mode_to_parameterized_unitary_value(self, mode):
        """
        Checks that add_mode_to_unitary function works correctly for a variety
        of positions, producing a value of 1 in the expected position.
        """
        unitary = ParameterizedUnitary(
            sp.Matrix([[6 * i + j for j in range(6)] for i in range(6)]), {}
        )
        new_unitary = add_mode_to_unitary(unitary, mode)
        # Check diagonal value on new mode
        assert new_unitary._unitary[mode, mode] == 1.0
        # Also confirm one off-diagonal value
        assert new_unitary._unitary[mode, mode - 1] == 0.0

    @pytest.mark.parametrize("mode", [0, 3, 5, 6])
    def test_add_mode_to_parameterized_unitary_diagonal(self, mode):
        """
        Checks that add_mode_to_unitary function works correctly for a variety
        of positions.
        """
        unitary = ParameterizedUnitary(
            sp.Matrix([[6 * i + j for j in range(6)] for i in range(6)]), {}
        )
        new_unitary = add_mode_to_unitary(unitary, mode)
        # Confirm preservation of unitary on diagonal
        assert (
            new_unitary._unitary[:mode, :mode] == unitary._unitary[:mode, :mode]
        )
        assert (
            new_unitary._unitary[mode + 1 :, mode + 1 :]
            == unitary._unitary[mode:, mode:]
        )

    @pytest.mark.parametrize("mode", [0, 3, 5, 6])
    def test_add_mode_to_parameterized_unitary_off_diagonal(self, mode):
        """
        Checks that add_mode_to_unitary function works correctly for a variety
        of positions.
        """
        unitary = ParameterizedUnitary(
            sp.Matrix([[6 * i + j for j in range(6)] for i in range(6)]), {}
        )
        new_unitary = add_mode_to_unitary(unitary, mode)
        # Confirm preservation on unitary off diagonal
        assert (
            new_unitary._unitary[mode + 1 :, :mode]
            == unitary._unitary[mode:, :mode]
        )
        assert (
            new_unitary._unitary[:mode, mode + 1 :]
            == unitary._unitary[:mode, mode:]
        )

    def test_add_heralds_to_state(self):
        """
        Tests that add heralds to state generates the correct mode structure.
        """
        s = [1, 2, 3, 4, 5]
        heralds = {1: 6, 6: 7}
        s_new = add_heralds_to_state(s, heralds)
        assert s_new == [1, 6, 2, 3, 4, 5, 7]

    def test_add_heralds_to_state_unordered(self):
        """
        Tests that add heralds to state generates the correct mode structure,
        when the heralding dict is not ordered.
        """
        s = [1, 2, 3, 4, 5]
        heralds = {6: 7, 1: 6}
        s_new = add_heralds_to_state(s, heralds)
        assert s_new == [1, 6, 2, 3, 4, 5, 7]

    def test_add_heralds_to_state_empty_herald(self):
        """
        Tests that the original state is returned when no heralds are used.
        """
        s = [randint(0, 5) for i in range(10)]
        s_new = add_heralds_to_state(s, {})
        assert s == s_new

    def test_add_heralds_to_state_no_modification(self):
        """
        Checks that add heralds to state does not modify the original state.
        """
        s = [randint(0, 5) for i in range(10)]
        s_copy = list(s)  # Creates copy of list
        add_heralds_to_state(s, {6: 7, 1: 6})
        assert s == s_copy

    def test_add_heralds_to_state_new_object(self):
        """
        Confirms that a new object is still created when no heralds are used
        with a given state.
        """
        s = [randint(0, 5) for i in range(10)]
        s_new = add_heralds_to_state(s, {})
        assert id(s) != id(s_new)

    @pytest.mark.parametrize("value", [1, 3, 2.0, None])
    def test_process_random_seed(self, value):
        """
        Confirms that process_random_seed allows valid values
        """
        process_random_seed(value)

    @pytest.mark.parametrize(
        "value", [0.5, 3.2, "1.1", "seed", [1], (1,), True, False]
    )
    def test_process_random_seed_invalid(self, value):
        """
        Confirms that check_random_seed detects invalid seeds.
        """
        with pytest.raises(TypeError):
            process_random_seed(value)

    @pytest.mark.parametrize(
        "value", [State([1, 0, 0, 1, 1, 0]), [1, 0, 0, 1, 1, 0]]
    )
    def test_valid_dual_rail_to_qubit(self, value):
        """
        Checks conversion from dual-rail to qubit states works correctly for
        valid values.
        """
        conv_state = convert.dual_rail_to_qubit(value)
        assert conv_state == State([0, 1, 0])

    @pytest.mark.parametrize(
        "value", [State([1, 0, 3, 1, 1, 0]), [1, 0, 3, 1, 1, 0]]
    )
    def test_allow_invalid_dual_rail_to_qubit(self, value):
        """
        Checks conversion from dual-rail to qubit states works correctly for
        valid values.
        """
        conv_state = convert.dual_rail_to_qubit(value, allow_invalid=True)
        assert conv_state == State([0, "X", 0])

    @pytest.mark.parametrize(
        "value",
        [
            State([1, 0, 0, 1, 1]),
            State([1, 1, 0, 1, 1, 0]),
            State([0, 0, 0, 1, 1, 0]),
            State([0, 2, 0, 1, 1, 0]),
            [0, 2, 0, 1, 1, 0],
        ],
    )
    def test_invalid_dual_rail_to_qubit(self, value):
        """
        Checks conversion from dual-rail to qubit raises a ValueError for
        incorrect values.
        """
        with pytest.raises(ValueError):
            convert.dual_rail_to_qubit(value)

    @pytest.mark.parametrize("value", [State([0, 1, 0]), [0, 1, 0], "010"])
    def test_valid_qubit_to_dual_rail(self, value):
        """
        Checks conversion from qubit to dual-rail states works correctly for
        valid values.
        """
        conv_state = convert.qubit_to_dual_rail(value)
        assert conv_state == State([1, 0, 0, 1, 1, 0])

    @pytest.mark.parametrize(
        "value", [State([0, 1, 2]), [-1, 1, 0], [0, 1.5, 1], "020"]
    )
    def test_invalid_qubit_to_dual_rail(self, value):
        """
        Checks conversion from qubit to dual-rail states works raises a
        ValueError for incorrect values.
        """
        with pytest.raises(ValueError):
            convert.qubit_to_dual_rail(value)

    def test_state_preservation(self):
        """
        Checks conversion to and from dual rail state to ensure the original
        state is always preserved
        """
        original_state = State([randint(0, 1) for _ in range(10)])
        conv_state = convert.qubit_to_dual_rail(original_state)
        recovered_state = convert.dual_rail_to_qubit(conv_state)
        assert original_state == recovered_state

    def test_threshold_mapping(self):
        """
        Checks threshold mapping produces the correct output.
        """
        original_state = State([1, 0, 2, 3, 0, 0, 1])
        conv_state = convert.threshold_mapping(original_state)
        assert conv_state == State([1, 0, 1, 1, 0, 0, 1])

    def test_threshold_mapping_invert(self):
        """
        Checks threshold mapping produces the correct output when inverting.
        """
        original_state = State([1, 0, 2, 3, 0, 0, 1])
        conv_state = convert.threshold_mapping(original_state, invert=True)
        assert conv_state == State([0, 1, 0, 0, 1, 1, 0])

    def test_parity_mapping(self):
        """
        Checks parity mapping produces the correct output.
        """
        original_state = State([1, 0, 2, 3, 0, 0, 1])
        conv_state = convert.parity_mapping(original_state)
        assert conv_state == State([1, 0, 0, 1, 0, 0, 1])

    def test_parity_mapping_invert(self):
        """
        Checks parity mapping produces the correct output when inverting.
        """
        original_state = State([1, 0, 2, 3, 0, 0, 1])
        conv_state = convert.parity_mapping(original_state, invert=True)
        assert conv_state == State([0, 1, 1, 0, 1, 1, 0])

    @pytest.mark.parametrize(
        "value", [0, 0.0, 0.5, 1, 1.0, Parameter(1), np.float64(0.67)]
    )
    def test_check_loss_valid(self, value):
        """
        Checks an error isn't raised when valid loss are values are provided.
        """
        check_loss(value)

    @pytest.mark.parametrize("value", [True, 0.6j, 1.5, 2, Parameter(2), "1"])
    def test_check_loss_invalid(self, value):
        """
        Checks an error is raised when invalid loss are values are provided.
        """
        with pytest.raises((TypeError, ValueError)):
            check_loss(value)


class TestPostSelection:
    """
    Unit tests for post-selection object.
    """

    def setup_method(self):
        """
        Creates a default state for testing.
        """
        self.test_state = State([1, 0, 1, 0, 1, 0])

    @pytest.mark.parametrize(
        "post_select",
        [
            PostSelection(),
            PostSelectionFunction(lambda s: True),  # noqa: ARG005
            DefaultPostSelection(),
        ],
    )
    def test_is_post_selection_type(self, post_select):
        """
        Checks all post-selection objects are child classes of the
        PostSelectionType.
        """
        assert isinstance(post_select, PostSelectionType)

    @pytest.mark.parametrize(
        ("modes", "photons"),
        [
            (0, 1),
            (0.0, 1.0),
            ((0,), (1,)),
            ((0.0,), (1.0,)),
            ((1, 0), 1),
            (0, (0, 1)),
            ((2, 4), (1, 2)),
        ],
    )
    def test_post_selection(self, modes, photons):
        """
        Tests a range of configuration of the test state and checks they all
        return True.
        """
        p = PostSelection()
        p.add(modes, photons)
        assert p.validate(self.test_state)

    @pytest.mark.parametrize(
        ("modes", "photons"),
        [(0, 0), ((0,), (0,)), ((1, 0), 2), (1, (1, 2)), ((2, 3), (0, 2))],
    )
    def test_post_selection_invalid(self, modes, photons):
        """
        Tests a range of configuration of the test state and checks they all
        return False.
        """
        p = PostSelection()
        p.add(modes, photons)
        assert not p.validate(self.test_state)

    def test_post_selection_duplicate(self):
        """
        Checks that by default post-selection does not allow for two rules on
        the same mode.
        """
        p = PostSelection()
        p.add(1, 1)
        with pytest.raises(ValueError):
            p.add(1, 2)

    def test_post_selection_duplicate_allowed(self):
        """
        Checks that post-selection allows for two rules on the same mode when
        multi_rules specified.
        """
        p = PostSelection(multi_rules=True)
        p.add(1, 1)
        p.add(1, 2)

    def test_post_selection_modes(self):
        """
        Checks that post-selection modes attribute is able to correctly return
        the correct values.
        """
        p = PostSelection()
        p.add(1, 1)
        assert p.modes == [1]
        p.add((2, 3), 1)
        assert p.modes == [1, 2, 3]
        p.add((4,), 1)
        assert p.modes == [1, 2, 3, 4]

    def test_post_selection_duplicates(self):
        """
        Checks that post-selection modes attribute is able to correctly return
        the correct values when a mode is used more than once
        """
        p = PostSelection(multi_rules=True)
        p.add(1, 1)
        assert p.modes == [1]
        p.add((2, 1), 1)
        assert p.modes == [1, 2]
        p.add((2,), 1)
        assert p.modes == [1, 2]

    @pytest.mark.parametrize(
        ("modes", "photons"),
        [
            (1, 1),
            ((1,), (1,)),
            ((1, 2), (1,)),
            ((1,), (1, 2)),
            ((2, 3), (1, 2)),
        ],
    )
    def test_post_selection_rules(self, modes, photons):
        """
        Checks that post-selection rules content is correct.
        """
        p = PostSelection()
        p.add(modes, photons)
        rules = p.rules
        if not isinstance(modes, tuple):
            modes = (modes,)
        if not isinstance(photons, tuple):
            photons = (photons,)
        assert rules[0].modes == modes
        assert rules[0].n_photons == photons

    def test_post_selection_no_rules(self):
        """
        Confirms post-selection object still functions when no rules have been
        provided.
        """
        p = PostSelection()
        assert p.validate(self.test_state)

    def test_post_selection_rules_length(self):
        """
        Confirms that rules property returns a list of the correct length.
        """
        p = PostSelection(multi_rules=True)
        p.add((0, 1), 2)
        p.add((0, 1), 2)
        p.add((2, 3), 1)
        assert len(p.rules) == 3

    @pytest.mark.parametrize("value", [-1, 0.5, ([1, 2],), "1.1"])
    def test_invalid_mode_values(self, value):
        """
        Confirms an error is raised if an invalid mode value is provided.
        """
        p = PostSelection()
        with pytest.raises((ValueError, TypeError)):
            p.add(value, 1)

    @pytest.mark.parametrize("value", [-1, 0.5, ([1, 2],), "1.1"])
    def test_invalid_photon_values(self, value):
        """
        Confirms an error is raised if an invalid photon value is provided.
        """
        p = PostSelection()
        with pytest.raises((ValueError, TypeError)):
            p.add(1, value)

    def test_default_post_selection_always_true(self):
        """
        Checks that default post-selection always returns true by sampling a
        large number of values.
        """
        ps = DefaultPostSelection()
        assert all(ps.validate(self.test_state) is True for _i in range(1000))

    @pytest.mark.parametrize("value", ["not_function", 1, [1, 2, 3]])
    def test_ps_function_enforced(self, value):
        """
        Checks an error is raised if a non-function type is passed to
        PostSelectionFunction.
        """
        with pytest.raises(TypeError):
            PostSelectionFunction(value)

    def test_ps_function_equivalance(self):
        """
        Checks that a function assigned to PostSelectionFunction will always
        return the same value as the function itself.
        """
        func = lambda s: s[1] + s[2] == 1 and s[3] == 0  # noqa: E731
        ps = PostSelectionFunction(func)
        # Check against known state
        assert func(self.test_state) == ps.validate(self.test_state)
        # Then randomly check for 100 states
        states = [[randint(0, 1) for _j in range(6)] for _i in range(100)]
        assert all(func(s) == ps.validate(s) for s in states)

    def test_rule(self):
        """
        Checks a rule can be created which implements a post-selection rule on a
        set of modes/photons.
        """
        Rule((1, 2), (3, 4))

    @pytest.mark.parametrize(
        ("modes", "photons"),
        [((0,), (1,)), ((1, 0), (1,)), ((0,), (0, 1)), ((2, 4), (1, 2))],
    )
    def test_rule_valid(self, modes, photons):
        """
        Desc
        """
        r = Rule(modes, photons)
        assert r.validate(self.test_state)

    def test_rule_tuple(self):
        """
        Checks rule as_tuple method returns expected value.
        """
        r = Rule((1, 2), (3, 4))
        assert r.as_tuple() == ((1, 2), (3, 4))


class TestSettings:
    """
    Set of unit tests for checking the settings module works as expected.
    """

    def test_valid_setting_assignment(self):
        """
        Checks the value of a valid setting can be modified.
        """
        settings.unitary_precision = 1

    def test_valid_setting_retrieval(self):
        """
        Checks the value of a valid setting can be modified and retrieved.
        """
        val = random()
        settings.unitary_precision = val
        assert settings.unitary_precision == val

    def test_invalid_setting_assignment(self):
        """
        Checks attempts to assign an invalid setting will raise an attribute
        error.
        """
        with pytest.raises(AttributeError):
            settings.test_setting = True

    def test_invalid_setting_retrieval(self):
        """
        Checks attempts to retrieve an invalid setting will raise an attribute
        error.
        """
        with pytest.raises(AttributeError):
            settings.test_setting  # noqa: B018

    def test_setting_in_str_repr(self):
        """
        Checks setting names when using str and repr on settings
        """
        assert "sampler_probability_threshold" in str(settings)
        assert "sampler_probability_threshold" in repr(settings)
