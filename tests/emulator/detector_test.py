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

import pytest

from lightworks import State
from lightworks.emulator import Detector


class TestDetector:
    """
    Unit tests to check behaviour of the Detector object in different
    situations.
    """

    def setup_method(self) -> None:  # noqa: D102
        self.lossy_detector = Detector(efficiency=0.1)  # Should be very lossy
        self.dc_detector = Detector(p_dark=0.1)  # High probability
        self.non_pnr_detector = Detector(photon_counting=False)

    def test_threshold_detection_multiphoton(self):
        """
        Checks that threshold detection behaves as expected for a state with
        multiple photons in a mode.
        """
        out = self.non_pnr_detector._get_output(State([0, 1, 2, 3, 0, 1]))
        assert out == State([0, 1, 1, 1, 0, 1])

    def test_threshold_detection_singlephoton(self):
        """
        Checks that threshold detection behaves as expected for a state with
        all modes having one or less photons.
        """
        out = self.non_pnr_detector._get_output(State([0, 1, 1, 1, 0, 0]))
        assert out == State([0, 1, 1, 1, 0, 0])

    def test_threshold_detection_zerostate(self):
        """
        Checks that threshold detection behaves as expected for the zero state.
        """
        out = self.non_pnr_detector._get_output(State([0, 0, 0, 0, 0]))
        assert out == State([0, 0, 0, 0, 0])

    def test_lossy_detector(self):
        """
        Confirms that the behaviour of detectors with imperfect detection
        efficiency is as expected.
        """
        measured = set()
        measured.update(
            self.lossy_detector._get_output(State([0, 2, 1, 0]))
            for _ in range(1000)
        )
        assert State([0, 1, 0, 0]) in measured

    def test_dark_counts(self):
        """Test that dark counts are working as expected."""
        measured = set()
        measured.update(
            self.dc_detector._get_output(State([0, 0, 0, 0]))
            for _ in range(1000)
        )
        n_photons = [s.n_photons for s in measured]
        assert max(n_photons) > 0

    def test_efficiency_modification(self):
        """
        Checks that the efficiency cannot be assigned to an invalid value.
        """
        detector = Detector()
        with pytest.raises(TypeError):
            detector.efficiency = True
        with pytest.raises(ValueError):
            detector.efficiency = -0.1
        with pytest.raises(ValueError):
            detector.efficiency = 1.1

    def test_p_dark_modification(self):
        """
        Checks that the p_dark cannot be assigned to an invalid value.
        """
        detector = Detector()
        with pytest.raises(TypeError):
            detector.p_dark = True
        with pytest.raises(ValueError):
            detector.p_dark = -0.1
        with pytest.raises(ValueError):
            detector.p_dark = 1.1

    def test_photon_counting_modification(self):
        """
        Checks that the photon_counting cannot be assigned to an invalid value.
        """
        detector = Detector()
        with pytest.raises(TypeError):
            detector.photon_counting = 0.5
        with pytest.raises(TypeError):
            detector.photon_counting = "True"

    def test_detector_random_seeding(self):
        """
        Checks repeatable results are produced when random seeding is used.
        """
        r_seed = 100 * random()
        detector = Detector(efficiency=0.5, p_dark=1e-2)
        n_samples = 100
        input_state = State([1, 0, 2, 1, 0, 1, 0])
        # Get original results
        first = []
        detector._set_random_seed(r_seed)
        for _i in range(n_samples):
            first.append(detector._get_output(input_state))
        # Reset seed and sample again
        second = []
        detector._set_random_seed(r_seed)
        for _i in range(n_samples):
            second.append(detector._get_output(input_state))
        # Check they are equivalent
        assert first == second
