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

"""
Class to simulate detector response when sampling, including detector
efficiency, dark counts and non-photon number resolving detectors.
"""

from numbers import Number
from random import random, seed

from ...sdk.state import State


class Detector:
    """
    Creates a detector which can be used to model the presence of imperfect
    detection properties when added to the Sampler.

    Args:

        efficiency (float, optional) : Set the per-channel efficiency of the
            detectors.

        p_dark (float, optional) : The probability of dark counts per detector,
            note that this will depend on the dark count rate and the system
            clock speed.

        photon_counting (bool, optional) : Set to True to use photon number
            resolving detectors and False to use threshold detection.

    """

    def __init__(
        self,
        efficiency: float = 1,
        p_dark: float = 0,
        photon_counting: bool = True,
    ) -> None:
        # Assign to attributes
        self.efficiency = efficiency
        self.p_dark = p_dark
        self.photon_counting = photon_counting

        return

    def __str__(self) -> str:
        return (
            f"Detector(efficiency = {self.efficiency}, p_dark = {self.p_dark},"
            f" photon_counting = {self.photon_counting})"
        )

    def __repr__(self) -> str:
        return "lightworks.emulator." + str(self)

    @property
    def efficiency(self) -> float:
        """The per-channel detection efficiency."""
        return self.__efficiency

    @efficiency.setter
    def efficiency(self, value: float) -> None:
        if not isinstance(value, Number) or isinstance(value, bool):
            raise TypeError("efficiency value should be numeric.")
        if not 0 <= value <= 1:
            raise ValueError("Value of efficiency should be in range [0,1].")
        self.__efficiency = value

    @property
    def p_dark(self) -> float:
        """The per-channel dark counts probability."""
        return self.__p_dark

    @p_dark.setter
    def p_dark(self, value: float) -> None:
        if not isinstance(value, Number) or isinstance(value, bool):
            raise TypeError("p_dark value should be numeric.")
        if not 0 <= value <= 1:
            raise ValueError("Value of p_dark should be in range [0,1].")
        self.__p_dark = value

    @property
    def photon_counting(self) -> float:
        """Controls whether the detectors are photon number resolving."""
        return self.__photon_counting

    @photon_counting.setter
    def photon_counting(self, value: float) -> None:
        if not isinstance(value, bool):
            raise TypeError("photon_counting should be a boolean.")
        self.__photon_counting = value

    def _get_output(self, in_state: State) -> State:
        """
        Sample an output state from the provided input.

        Args:

            in_state (State) : The input state to the detection module.

        Returns:

            State: The processed output state.

        """
        # If detectors are perfect then just return input
        if self.efficiency == 1 and self.p_dark == 0 and self.photon_counting:
            return in_state
        # Convert state to list
        output = list(in_state)
        # Account for efficiency
        if self.efficiency < 1:
            for mode, n in enumerate(in_state):
                for _i in range(n):
                    if random() > self.efficiency:
                        output[mode] -= 1
        # Then include dark counts
        if self.p_dark > 0:
            for mode in range(len(in_state)):
                if random() < self.p_dark:
                    output[mode] += 1
        # Also account for non-photon counting detectors
        if not self.photon_counting:
            output = [1 if count >= 1 else 0 for count in output]

        return State(output)

    def _set_random_seed(self, r_seed: float | None) -> None:
        """
        Set the random seed for the detector to produce repeatable results.
        """
        seed(r_seed)
