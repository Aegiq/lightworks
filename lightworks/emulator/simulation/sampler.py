# Copyright 2024 - 2025 Aegiq Ltd.
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

from collections import Counter
from collections.abc import Callable

import numpy as np

from lightworks.emulator.components import Detector, Source
from lightworks.sdk.circuit.photonic_compiler import CompiledPhotonicCircuit
from lightworks.sdk.results import SamplingResult
from lightworks.sdk.state import State
from lightworks.sdk.tasks import SamplerTask
from lightworks.sdk.utils.exceptions import SamplerError
from lightworks.sdk.utils.heralding import (
    add_heralds_to_state,
    remove_heralds_from_state,
)
from lightworks.sdk.utils.post_selection import (
    DefaultPostSelection,
    PostSelectionType,
)
from lightworks.sdk.utils.random import process_random_seed

from .probability_distribution import pdist_calc
from .runner import RunnerABC


class SamplerRunner(RunnerABC):
    """
    Calculates the output probability distribution for a configuration and
    finds produces a set of N samples from this.

    Args:

        data (SamplerTask) : The task which is to be executed.

        pdist_function (Callable) : Function for calculating probability
            distributions for a provided unitary & input.

    Attributes:

        source (Source) : The in-use Source object. If the source in the data
            was originally set to None then this a new default Source object is
            created.

        detector (Detector) : The in-use Detector object. If the detector in the
            data was originally set to None then this a new default Detector
            object is created.

    """

    def __init__(
        self,
        data: SamplerTask,
        pdist_function: Callable[
            [CompiledPhotonicCircuit, State], dict[State, float]
        ],
    ) -> None:
        self.data = data
        self.source = Source() if self.data.source is None else self.data.source
        self.detector = (
            Detector() if self.data.detector is None else self.data.detector
        )
        self.pdist_function = pdist_function

    def distribution_calculator(self) -> dict[State, float]:
        """
        Calculates the output probability distribution for the provided
        configuration. This needs to be done before sampling.
        """
        # Check circuit and input modes match
        if self.data.circuit.input_modes != len(self.data.input_state):
            raise ValueError(
                "Mismatch in number of modes between input and circuit."
            )
        # Add heralds to the included input
        modified_state = add_heralds_to_state(
            self.data.input_state, self.data.circuit.heralds.input
        )
        input_state = State(modified_state)
        # Then build with source
        all_inputs = self.source._build_statistics(input_state)
        # And find probability distribution
        pdist = pdist_calc(self.data.circuit, all_inputs, self.pdist_function)
        # Special case to catch an empty distribution
        if not pdist:
            pdist = {State([0] * self.data.circuit.n_modes): 1}
        # Assign calculated distribution to attribute
        self.probability_distribution = pdist
        herald_modes = list(self.data.circuit.heralds.output.keys())
        self.herald_cache = _HeraldCache(herald_modes)
        return pdist

    def run(self) -> SamplingResult:
        """
        Performs sampling using the calculated probability distribution.

        Returns:

            SamplingResult : A dictionary containing the different output
                states and the number of counts for each one.

        """
        if not hasattr(self, "probability_distribution"):
            raise RuntimeError(
                "Probability distribution has not been calculated. This likely "
                "results from an error in Lightworks."
            )

        post_selection = (
            DefaultPostSelection()
            if self.data.post_selection is None
            else self.data.post_selection
        )
        min_detection = (
            0 if self.data.min_detection is None else self.data.min_detection
        )

        if self.data.sampling_mode == "input":
            return self._sample_N_inputs(
                self.data.n_samples,
                post_selection,
                min_detection,
                self.data.random_seed,
            )
        return self._sample_N_outputs(
            self.data.n_samples,
            post_selection,
            min_detection,
            self.data.random_seed,
        )

    def _sample_N_inputs(  # noqa: N802
        self,
        N: int,  # noqa: N803
        post_select: PostSelectionType,
        min_detection: int = 0,
        seed: int | None = None,
    ) -> SamplingResult:
        """
        Function to sample from the configured system by running N clock cycles
        of the system. In each of these clock cycles the input may differ from
        the target input, dependent on the source properties, and there may be
        a number of imperfections in place which means that photons are not
        measured or false detections occur. This means it is possible to for
        less than N measured states to be returned.

        Args:

            N (int) : The number of samples to take from the circuit.

            post_select (PostSelection) : A PostSelection object or function
                which applies a provided set of post-selection criteria to a
                state.

            min_detection (int, optional) : Post-select on a given minimum
                total number of photons, this should not include any heralded
                photons.

            seed (int|None, optional) : Option to provide a random seed to
                reproducibly generate samples from the function. This is
                optional and can remain as None if this is not required.

        Returns:

            SamplingResult : A dictionary containing the different output
                states and the number of counts for each one.

        """
        try:
            samples = self._gen_samples_from_dist(
                self.probability_distribution, N, seed, normalize=False
            )
        # Sometimes the probability distribution will not quite be normalized,
        # in this case try to re-normalize it.
        except ValueError as e:
            total_p = sum(self.probability_distribution.values())
            if abs(total_p - 1) > 0.01:
                msg = (
                    "Probability distribution significantly deviated from "
                    f"required normalisation ({total_p})."
                )
                raise ValueError(msg) from e
            samples = self._gen_samples_from_dist(
                self.probability_distribution, N, seed, normalize=True
            )
            self.probability_distribution = {
                k: v / total_p for k, v in self.probability_distribution.items()
            }
        filtered_samples = []
        # Get heralds and pre-calculate items
        heralds = self.data.circuit.heralds.output
        if (
            heralds
            and max(heralds.values()) > 1
            and not self.detector.photon_counting
        ):
            raise SamplerError(
                "Non photon number resolving detectors cannot be used when"
                "a heralded mode has more than 1 photon."
            )
        herald_items = list(heralds.items())
        # Set detector seed before sampling
        self.detector._set_random_seed(seed)
        # Process output states
        for state, count in samples.items():
            for _ in range(count):
                output_state = self.detector._get_output(state)
                # Checks herald requirements are met
                for m, n in herald_items:
                    if output_state[m] != n:
                        break
                # If met then remove heralded modes and store
                else:
                    herald_state = (
                        self.herald_cache[output_state]
                        if heralds
                        else output_state
                    )
                    if (
                        post_select.validate(herald_state)
                        and herald_state.n_photons >= min_detection
                    ):
                        filtered_samples.append(herald_state)
        counted = dict(Counter(filtered_samples))
        return SamplingResult(counted, self.data.input_state)

    def _sample_N_outputs(  # noqa: N802
        self,
        N: int,  # noqa: N803
        post_select: PostSelectionType,
        min_detection: int = 0,
        seed: int | None = None,
    ) -> SamplingResult:
        """
        Function to generate N output samples from a system, according to a set
        of selection criteria. The function will raise an error if the
        selection criteria is too strict and removes all outputs. Also note
        this cannot be used to simulate detector dark counts.

        Args:

            N (int) : The number of samples that are to be returned.

            post_select (PostSelection) : A PostSelection object or function
                which applies a provided set of post-selection criteria to a
                state.

            min_detection (int, optional) : Post-select on a given minimum
                total number of photons, this should not include any heralded
                photons.

            seed (int|None, optional) : Option to provide a random seed to
                reproducibly generate samples from the function. This is
                optional and can remain as None if this is not required.

        Returns:

            SamplingResult : A dictionary containing the different output
                states and the number of counts for each one.

        """
        pdist = self.probability_distribution
        if self.detector.p_dark > 0 or self.detector.efficiency < 1:
            raise SamplerError(
                "To use detector dark counts or sub-unity detector efficiency "
                "the sampling mode must be set to 'input'."
            )
        # Get heralds and pre-calculate items
        heralds = self.data.circuit.heralds.output
        if (
            heralds
            and max(heralds.values()) > 1
            and not self.detector.photon_counting
        ):
            raise SamplerError(
                "Non photon number resolving detectors cannot be used when"
                "a heralded mode has more than 1 photon."
            )
        herald_items = list(heralds.items())
        # Convert distribution using provided data
        new_dist: dict[State, float] = {}
        for s, p in pdist.items():
            # Apply threshold detection
            if not self.detector.photon_counting:
                s = State([min(i, 1) for i in s])  # noqa: PLW2901
            # Check heralds
            for m, n in herald_items:
                if s[m] != n:
                    break
            else:
                # Then remove herald modes
                herald_state = self.herald_cache[s] if heralds else s
                # Check state meets min detection and post-selection criteria
                # across remaining modes
                if (
                    herald_state.n_photons >= min_detection
                    and post_select.validate(herald_state)
                ):
                    if herald_state in new_dist:
                        new_dist[herald_state] += p
                    else:
                        new_dist[herald_state] = p
        pdist = new_dist
        # Check some states are found
        if not pdist:
            raise SamplerError(
                "No output states compatible with provided post-selection/"
                "min-detection criteria."
            )
        # Then generate samples and return
        return SamplingResult(
            self._gen_samples_from_dist(pdist, N, seed, normalize=True),
            self.data.input_state,
        )

    def _gen_samples_from_dist(
        self,
        distribution: dict[State, float],
        n_samples: int,
        seed: int | None,
        normalize: bool = False,
    ) -> dict[State, int]:
        """
        Takes a computed probability distribution and generates the request
        number of samples, returning this as a dictionary of states and counts.
        """
        # Re-normalise distribution probabilities
        mapping = dict(enumerate(distribution))
        probs = np.array(list(distribution.values()), dtype=float)
        if normalize:
            probs /= sum(probs)
        # Generate N random samples and then process and count output states
        rng = np.random.default_rng(process_random_seed(seed))
        samples = rng.choice(range(len(mapping)), p=probs, size=n_samples)
        # Count states and convert to results object
        return {mapping[n]: c for n, c in Counter(samples).items()}


class _HeraldCache:
    """
    Used for reducing the number of repeated computations of states with
    heralded modes removed.
    """

    def __init__(self, herald_modes: list[int]) -> None:
        self.cache: dict[State, State] = {}
        self.herald_modes = list(herald_modes)

    def __getitem__(self, state: State) -> State:
        """
        Checks if a state is in the cache, otherwise removes the heralded modes.
        """
        if state not in self.cache:
            self.cache[state] = State(
                remove_heralds_from_state(state, self.herald_modes)
            )
        return self.cache[state]
