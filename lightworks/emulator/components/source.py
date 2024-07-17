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
Models a non-ideal source of single photons, it can be used to build input
state variations, accounting for a non-ideal brightness, impurity in the single
photon stream and indistinguishability between different photons.
"""

from numbers import Number

from ...sdk.state import State
from ..state import AnnotatedState


class Source:
    """
    Simulates an imperfect single photon source to model properties of photon
    inputs in the Sampler.

    Args:

        purity (float) : The purity of the photon stream from the single photon
            source, should be given as a decimal. Note that it is assumed that
            any additional photons are distinguishable from the source photons.

        brightness (float) : The brightness of the single photon source as a
            decimal value.

        indistinguishability (float) : The indistinguishability of the input
            photons.

        probability_threshold (float, optional) : Set a probability threshold
            for the input, this can be used to remove any inputs with
            probabilities below this value. Note, this should be used with some
            caution as it may reduce the overall accuracy of the simulation.

    """

    def __init__(
        self,
        purity: float = 1,
        brightness: float = 1,
        indistinguishability: float = 1,
        probability_threshold: float = 0,
    ) -> None:
        self.purity = purity
        self.brightness = brightness
        self.indistinguishability = indistinguishability
        self.probability_threshold = probability_threshold

        return

    def __str__(self) -> str:
        return (
            f"Source(purity = {self.purity}, brightness = {self.brightness}, "
            f"indistinguishability = {self.indistinguishability}, "
            f"probability threshold = {self.probability_threshold})"
        )

    def __repr__(self) -> str:
        return "lightworks.emulator." + str(self)

    @property
    def brightness(self) -> float:
        """The output efficiency of the single photon source."""
        return self.__brightness

    @brightness.setter
    def brightness(self, value: float) -> None:
        quantity_check(value, "brightness")
        self.__brightness = value

    @property
    def purity(self) -> float:
        """The purity as a source, given as a decimal. Purity = 1 - g2."""
        return self.__purity

    @purity.setter
    def purity(self, value: float) -> None:
        # Special check for purity as it requires value >= 0.5
        if not 0.5 < value <= 1:
            raise ValueError("Value of purity should be in range (0.5,1].")
        quantity_check(value, "purity")
        self.__purity = value

    @property
    def indistinguishability(self) -> float:
        """The measured indistinguishability of the single photon source."""
        return self.__indistinguishability

    @indistinguishability.setter
    def indistinguishability(self, value: float) -> None:
        quantity_check(value, "indistinguishability")
        self.__indistinguishability = value

    @property
    def probability_threshold(self) -> float:
        """
        Define a probability threshold below which single input states are
        discarded.
        """
        return self.__probability_threshold

    @probability_threshold.setter
    def probability_threshold(self, value: float) -> None:
        quantity_check(value, "probability_threshold")
        self.__probability_threshold = value

    def check_number(self, state: State) -> int:
        """
        Check the number of possible inputs states that a given target input
        and source properties will produce.

        Args:

            state (State) : The target input state produced by the source.

        Returns:

            int : The number of states contained in the generated source input
                state probability distribution.

        """
        return len(self._build_statistics(state))

    def _build_statistics(
        self, state: State
    ) -> dict[State, float] | dict[AnnotatedState, float]:
        """
        Determine the true input statistics for a given target input and the
        provided properties of the single photon source.

        Args:

            state (State) : The target input state for the calculation.

        Returns:

            dict : The created statistics dictionary containing all possible
                inputs and their probabilities. This returned inputs will
                either be States or AnnotatedStates depending on the required
                input properties.

        """
        # Vary the method used depending on the parameter type.
        if self.purity == 1 and self.indistinguishability == 1:
            stats_dict = self._build_statistics_basic(state)
        else:
            stats_dict = self._build_statistics_full(state)

        # Apply probability thresholding if required
        if self.probability_threshold:
            thresholded_dict = {
                s: p
                for s, p in stats_dict.items()
                if p >= self.probability_threshold
            }
            # Re-normalize after removing values
            total = sum(thresholded_dict.values())
            for s, p in thresholded_dict.items():
                thresholded_dict[s] = p / total
            stats_dict = thresholded_dict

        return stats_dict

    def _build_statistics_basic(self, state: State) -> dict:
        """
        Build a basic version of statistics which only accounts for source
        brightness.

        Args:

            state (State) : The target input state for the calculation.

        Returns:

            dict : The created statistics dictionary containing all possible
                inputs and their probabilities.

        """
        n_modes = len(state)
        stats: list[tuple[float, list[int]]] = []
        # Loop over each mode
        for mode, count in enumerate(state):
            if not count:
                continue
            for _i in range(count):
                # For a given photon in a mode work out all possible input
                # combinations
                sub_s = [
                    (
                        self.brightness,
                        [1 if mode == j else 0 for j in range(n_modes)],
                    )
                ]
                # Add zero mode from less than perfect brightness
                if self.brightness < 1:
                    sub_s.append((1 - self.brightness, [0] * n_modes))
                # Create stats list on first run
                if not stats:
                    stats = sub_s
                # Then combine stats with new calculated states to find new
                # probabilities
                else:
                    new_stats = []
                    for p1, s1 in stats:
                        for p2, s2 in sub_s:
                            new_stats.append(
                                (p1 * p2, [x + y for x, y in zip(s1, s2)])
                            )
                    stats = new_stats
        # Combine any duplicate results
        stats_dict: dict[State, float] = {}  # Convert stats to dict
        for probs, s in stats:
            s_conv = State(s)
            if s_conv not in stats_dict:
                stats_dict[s_conv] = probs
            else:
                stats_dict[s_conv] += probs
        # Catch any empty returns and give input
        if not stats_dict:
            stats_dict[state] = 1.0
        return stats_dict

    def _build_statistics_full(self, state: State) -> dict:
        """
        Builds a full version of statistics which can account for the
        brightness, indistinguishability and purity of the source.

        Args:

            state (State) : The target input state for the calculation.

        Returns:

            dict : The created statistics dictionary containing all possible
                inputs and their probabilities. All inputs will be represented
                as annotated states, which use labelling to denote photons
                which are distinguishable from each other.

        """
        # Reset counter
        self._counter = 1
        # Get the full distribution and then perform remapping
        stats_dict = self._full_distribution(state)
        return self._remap_distribution(stats_dict)

    def _remap_distribution(self, distribution: dict) -> dict:
        """
        Remaps a provided distribution to a common form, enabling equivalent
        entries to be removed.
        """
        # Remap states to be correctly ordered
        new_dist = {}
        for state, p in distribution.items():
            # Determine all labels in a given state and create a mapping which
            # converts all labels to their minimum value
            all_labels = []
            for mode in state:
                all_labels += mode
            all_labels = list(dict.fromkeys(all_labels))
            mapping = {all_labels[i]: i for i in range(len(all_labels))}
            # Apply determined mapping to each state
            state = state.s  # noqa: PLW2901
            for j, mode in enumerate(state):
                for i, m in enumerate(mode):
                    mode[i] = mapping[m]
                state[j] = mode
            state = AnnotatedState(state)  # noqa: PLW2901
            # Add new state to new distribution
            if state not in new_dist:
                new_dist[state] = p
            else:
                new_dist[state] += p

        return new_dist

    def _full_distribution(self, state: State) -> dict:
        """
        Calculates the full input state probability distribution for a given
        state.
        """
        # Find groups of empty modes
        to_group, to_skip = group_empty_modes(state)
        input_dist: dict[AnnotatedState, float] = {}
        # Loop over each mode and find distribution
        for i, n in enumerate(state):
            if i in to_skip:
                continue
            # Added groups of empty photons
            if i in to_group:
                if not input_dist:
                    input_dist = {AnnotatedState([[]] * len(to_group[i])): 1}
                else:
                    input_dist = {
                        s1 + AnnotatedState([[]] * len(to_group[i])): p1
                        for s1, p1 in input_dist.items()
                    }
            else:
                calc_dist = self._single_mode_distribution(n)
                # Then combine with the full input distribution
                if not input_dist:
                    input_dist = calc_dist
                else:
                    new_dist = {}
                    for s1, p1 in input_dist.items():
                        for s2, p2 in calc_dist.items():
                            new_dist[s1 + s2] = p1 * p2
                    input_dist = new_dist

        return input_dist

    def _single_mode_distribution(self, n_photons: int) -> dict:
        """
        Finds the input state statistics for a number of photons on a given
        mode.
        """
        # Check for special case in which a mode is empty
        if n_photons == 0:
            return {AnnotatedState([[]]): 1}
        mode_dist: list[tuple[list, float]] = []
        # Get distribution for each photon and combine with mode distribution
        for _i in range(n_photons):
            calc_dist = self._single_photon_distribution()
            if mode_dist == []:
                mode_dist = calc_dist
            else:
                new_dist = [
                    (d1[0] + d2[0], d1[1] * d2[1])
                    for d1 in mode_dist
                    for d2 in calc_dist
                ]
                mode_dist = new_dist
        # Sort labels for easier processing
        for d in mode_dist:
            d[0].sort()
        # Convert to annotated states and add to a dictionary
        dist_dict = {}
        for s, p in mode_dist:
            state = AnnotatedState([s])
            if state not in dist_dict:
                dist_dict[state] = p
            else:
                dist_dict[state] += p
        return dist_dict

    def _single_photon_distribution(self) -> list[tuple[list, float]]:
        """
        Calculates the distribution for a single photon with imperfect
        properties.
        """
        # Define some useful quantities
        nu = self.brightness
        p_i = self.indistinguishability**0.5
        p_d = 1 - p_i
        # Find pure state probability using purity
        p1 = purity_to_prob(self.purity)
        p2 = 1 - p1
        # Define all required probabilities
        c0 = 1 - nu * (p1 + p2 * nu + 2 * (1 - nu) * p2)
        c1 = p_i * nu * (p1 + (1 - nu) * p2)
        c1d = p_d * nu * (p1 + (1 - nu) * p2)
        c1dp = nu * (1 - nu) * p2
        c12d = nu**2 * p_i * p2
        c1d2d = nu**2 * p_d * p2
        # Determine unique labels for distinguishable and multi-photon photons
        dpc = self._counter
        mpc = self._counter + 1
        self._counter += 2
        # Loop through all entries and add if each probability is non-zero
        to_add: list[tuple[float, list]] = [
            (c0, []),
            (c1, [0]),
            (c1d, [dpc]),
            (c1dp, [mpc]),
            (c12d, [0, mpc]),
            (c1d2d, [dpc, mpc]),
        ]
        out: list[tuple[list, float]] = []
        for p, s in to_add:
            if p > 0:
                out += [(s, p)]

        return out


def group_empty_modes(state: State) -> tuple[dict, list]:
    """
    Used to find groups of empty modes, this can be used to speed up the source
    distribution calculation.
    """
    to_group = {}
    to_skip = []
    # Loop over each mode in the state
    for i, s in enumerate(state):
        # Skip any modes already grouped or the last mode
        if i in to_skip or i == len(state) - 1:
            continue
        if s == 0:
            # If only a single empty mode then skip
            if state[i + 1] > 0:  # type: ignore
                continue
            # Find how many empty modes there are
            n = 1
            while state[i + n] == 0:
                n += 1
                if i + n > len(state) - 1:
                    break
            to_group[i] = list(range(i, i + n))
            to_skip += list(range(i + 1, i + n))

    return (to_group, to_skip)


def purity_to_prob(purity: float) -> float:
    """
    Return single photon probability for a given purity, probability of a two
    photon state will be twice this.
    """
    if purity <= 0.5:
        raise ValueError("Purity values below 0.5 not supported.")
    if purity < 1:
        g2 = 1 - purity
        b = 2 * (1 - (1 / g2))
        p1 = 1 - (-b - (b**2 - 4) ** 0.5) / 2
    else:
        p1 = 1
    return p1


def quantity_check(value: float, name: str) -> None:
    """
    General function to confirm that a quantity is numerical and lies in the
    range [0,1].

    Args:

        value (float) : The quantity that is to be checked.

        name (str) : A name to use for the quantity when displaying error
            messages.

    Raises:

        TypeError : When quantity isn't numeric.

        ValueError : When quantity is not within the range [0,1].

    """
    if not isinstance(value, Number) or isinstance(value, bool):
        msg = f"{name} should be a numerical value."
        raise TypeError(msg)
    if not 0 <= value <= 1:
        msg = f"Value of {name} should be in range [0,1]."
        raise ValueError(msg)
