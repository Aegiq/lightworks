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
        self.func = pdist_function

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
        pdist = pdist_calc(self.data.circuit, all_inputs, self.func)
        # Special case to catch an empty distribution
        if not pdist:
            pdist = {State([0] * self.data.circuit.n_modes): 1}
        # Assign calculated distribution to attribute
        self.probability_distribution = pdist
        herald_modes = list(self.data.circuit.heralds.output.keys())
        self.full_to_heralded = {
            s: State(remove_heralds_from_state(s, herald_modes)) for s in pdist
        }
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
        pdist = self.probability_distribution
        vals = np.zeros(len(pdist), dtype=object)
        for i, k in enumerate(pdist.keys()):
            vals[i] = k
        # Generate N random samples and then process and count output states
        rng = np.random.default_rng(process_random_seed(seed))
        try:
            samples = rng.choice(vals, p=list(pdist.values()), size=N)
        # Sometimes the probability distribution will not quite be normalized,
        # in this case try to re-normalize it.
        except ValueError as e:
            total_p = sum(pdist.values())
            if abs(total_p - 1) > 0.01:
                msg = (
                    "Probability distribution significantly deviated from "
                    f"required normalisation ({total_p})."
                )
                raise ValueError(msg) from e
            norm_p = [p / total_p for p in pdist.values()]
            samples = rng.choice(vals, p=norm_p, size=N)
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
        herald_modes = list(heralds.keys())
        herald_items = list(heralds.items())
        # Set detector seed before sampling
        self.detector._set_random_seed(seed)
        # Process output states
        for state in samples:
            state = self.detector._get_output(state)  # noqa: PLW2901
            # Checks herald requirements are met
            for m, n in herald_items:
                if state[m] != n:
                    break
            # If met then remove heralded modes and store
            else:
                if heralds:
                    if state not in self.full_to_heralded:
                        self.full_to_heralded[state] = State(
                            remove_heralds_from_state(state, herald_modes)
                        )
                    hs = self.full_to_heralded[state]
                else:
                    hs = state
                if post_select.validate(hs) and hs.n_photons >= min_detection:
                    filtered_samples.append(hs)
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
        herald_modes = list(heralds.keys())
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
                if heralds:
                    if s not in self.full_to_heralded:
                        self.full_to_heralded[s] = State(
                            remove_heralds_from_state(s, herald_modes)
                        )
                    new_s = self.full_to_heralded[s]
                else:
                    new_s = s
                # Check state meets min detection and post-selection criteria
                # across remaining modes
                if new_s.n_photons >= min_detection and post_select.validate(
                    new_s
                ):
                    if new_s in new_dist:
                        new_dist[new_s] += p
                    else:
                        new_dist[new_s] = p
        pdist = new_dist
        # Check some states are found
        if not pdist:
            raise SamplerError(
                "No output states compatible with provided post-selection/"
                "min-detection criteria."
            )
        # Re-normalise distribution probabilities
        probs = np.array(list(pdist.values()), dtype=float)
        probs /= sum(probs)
        # Put all possible states into array
        vals = np.zeros(len(pdist), dtype=object)
        for i, k in enumerate(pdist.keys()):
            vals[i] = k
        # Generate N random samples and then process and count output states
        rng = np.random.default_rng(process_random_seed(seed))
        samples = rng.choice(vals, p=probs, size=N)
        # Count states and convert to results object
        counted = dict(Counter(samples))
        return SamplingResult(counted, self.data.input_state)
