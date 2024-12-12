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
from lightworks.emulator import Source
from lightworks.emulator.state import AnnotatedState


class TestSource:
    """
    Unit tests to check behaviour and statistics returned while using the
    Source object of the emulator.
    """

    def test_perfect_input(self):
        """Checks only one state is returned when a perfect input is used."""
        source = Source()
        stats = source._build_statistics(State([1, 0, 1, 0]))
        assert stats == {State([1, 0, 1, 0]): 1}

    def test_imperfect_brightness(self):
        """Checks the return values when an imperfect brightness is used."""
        source = Source(brightness=0.915)
        stats = source._build_statistics(State([1, 0, 1, 0]))
        stats = {s: round(p, 6) for s, p in stats.items()}
        expected = {
            State([1, 0, 1, 0]): 0.837225,
            State([1, 0, 0, 0]): 0.077775,
            State([0, 0, 1, 0]): 0.077775,
            State([0, 0, 0, 0]): 0.007225,
        }
        assert stats == expected

    def test_imperfect_source(self):
        """
        Checks return values are correct when all possible imperfections in a
        source are used. Also, confirms all states returned are of the correct
        length.
        """
        source = Source(
            brightness=0.493,
            purity=0.984,
            indistinguishability=0.937,
            probability_threshold=0,
        )
        stats = source._build_statistics(State([0, 0, 2, 1, 0, 1, 0, 0]))
        # Check number of inputs is correct
        assert len(stats) == 204
        # Check some random values
        assert stats[
            AnnotatedState([[], [], [0], [0], [], [0], [], []])
        ] == pytest.approx(0.10844531898317525, 1e-6)
        assert stats[AnnotatedState([[]] * 8)] == pytest.approx(
            0.06502113537379095, 1e-6
        )
        # Ensure all state lengths are correct
        for state in stats:
            assert len(state) == 8

    def test_source_thresholding(self):
        """
        Confirms correct behaviour when probability thresholding is applied to
        the Source object.
        """
        source = Source(
            brightness=0.493,
            purity=0.984,
            indistinguishability=0.937,
            probability_threshold=1e-6,
        )
        stats = source._build_statistics(State([0, 0, 2, 1, 0, 1, 0, 0]))
        # Check length is as expected after thresholding
        assert len(stats) == 89

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            (True, TypeError),
            ("0.7", TypeError),
            (0.5, ValueError),
            (1.1, ValueError),
        ],
    )
    def test_purity_modification(self, value, error):
        """
        Checks that the purity cannot be assigned to an invalid value.
        """
        source = Source()
        with pytest.raises(error):
            source.purity = value

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            (True, TypeError),
            ("0.5", TypeError),
            (-0.1, ValueError),
            (1.1, ValueError),
        ],
    )
    def test_brightness_modification(self, value, error):
        """
        Checks that the brightness cannot be assigned to an invalid value.
        """
        source = Source()
        with pytest.raises(error):
            source.brightness = value

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            (True, TypeError),
            ("0.5", TypeError),
            (-0.1, ValueError),
            (1.1, ValueError),
        ],
    )
    def test_indistinguishability_modification(self, value, error):
        """
        Checks that the indistinguishability cannot be assigned to an invalid
        value.
        """
        source = Source()
        with pytest.raises(error):
            source.indistinguishability = value

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            (True, TypeError),
            ("0.5", TypeError),
            (-0.1, ValueError),
            (1.1, ValueError),
        ],
    )
    def test_probability_threshold_modification(self, value, error):
        """
        Checks that the probability_threshold cannot be assigned to an invalid
        value.
        """
        source = Source()
        with pytest.raises(error):
            source.probability_threshold = value
