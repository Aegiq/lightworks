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

from ...emulator.backends.fock_backend import FockBackend
from ...emulator.components import Detector, Source
from ...emulator.utils.probability_distribution import pdist_calc
from ...sdk.utils.post_selection import PostSelectionType
from ..circuit import Circuit
from ..results import SamplingResult
from ..state import State
from ..utils import (
    ModeMismatchError,
    SamplerError,
    add_heralds_to_state,
    process_post_selection,
    process_random_seed,
    remove_heralds_from_state,
)
from .task import Task


class Sampler(Task):
    """
    Used to randomly sample from the photon number distribution output of a
    provided circuit. The distribution is calculated when the class is first
    called and then used to return output states with the sample function. Also
    supports the inclusion of imperfect source and detection properties.

    Args:

        circuit : The circuit to sample output states from.

        input_state (State) : The input state to use with the circuit for
            sampling.

        n_samples (int) : The number of samples that are to be returned.

        source (Source, optional) : Provide a source object to simulate an
            imperfect input. This defaults to None, which will create a perfect
            source.

        detector (Detector, optional) : Provide detector to simulate imperfect
            detector probabilities. This defaults to None, which will assume a
            perfect detector.

        post_select (PostSelection | function, optional) : A PostSelection
            object or function which applies a provided set of
            post-selection criteria to a state.

        min_detection (int, optional) : Post-select on a given minimum
            total number of photons, this should not include any heralded
            photons.

        random_seed (int|None, optional) : Option to provide a random seed to
            reproducibly generate samples from the function. This is
            optional and can remain as None if this is not required.

        sampling_mode (str, optional) : Sets the mode of the Sampler. In input
            mode, N cycles of the system are measured, and only results which
            meet any assigned criteria are returned. In output mode, N valid
            samples are produced from the system. Should be either 'input' or
            'output', defaults to 'output'.

    """

    __compatible_backends__ = ("permanent", "slos")

    def __init__(
        self,
        circuit: Circuit,
        input_state: State,
        n_samples: int,
        post_selection: PostSelectionType | Callable | None = None,
        min_detection: int = 0,
        source: Source | None = None,
        detector: Detector | None = None,
        random_seed: int | None = None,
        sampling_mode: str = "output",
    ) -> None:
        # Assign provided quantities to attributes
        self.circuit = circuit
        self.input_state = input_state
        self.source = source  # type: ignore[assignment]
        self.detector = detector  # type: ignore[assignment]
        self.n_samples = n_samples
        self.post_selection = post_selection  # type: ignore[assignment]
        self.min_detection = min_detection
        self.random_seed = random_seed
        self.sampling_mode = sampling_mode

    @property
    def circuit(self) -> Circuit:
        """
        Stores the circuit to be used for simulation, should be a Circuit
        object.
        """
        return self.__circuit

    @circuit.setter
    def circuit(self, value: Circuit) -> None:
        if not isinstance(value, Circuit):
            raise TypeError(
                "Provided circuit should be a Circuit or Unitary object."
            )
        self.__circuit = value

    @property
    def input_state(self) -> State:
        """The input state to be used for sampling."""
        return self.__input_state

    @input_state.setter
    def input_state(self, value: State) -> None:
        if not isinstance(value, State):
            raise TypeError("A single input of type State should be provided.")
        if len(value) != self.circuit.input_modes:
            raise ModeMismatchError(
                "Incorrect input length for provided circuit."
            )
        # Also validate state values
        value._validate()
        self.__input_state = value

    @property
    def source(self) -> Source:
        """
        Details the properties of the Source used for creation of inputs to
        the Sampler.
        """
        return self.__source

    @source.setter
    def source(self, value: Source) -> None:
        if value is None:
            value = Source()
        if not isinstance(value, Source):
            raise TypeError("Provided source should be a Source object.")
        self.__source = value

    @property
    def detector(self) -> Detector:
        """
        Details the properties of the Detector used for photon measurement.
        """
        return self.__detector

    @detector.setter
    def detector(self, value: Detector | None) -> None:
        if value is None:
            value = Detector()
        if not isinstance(value, Detector):
            raise TypeError("Provided detector should be a Detector object.")
        self.__detector = value

    @property
    def n_samples(self) -> int:
        """Stores the number of samples to be collected in an experiment."""
        return self.__n_samples

    @n_samples.setter
    def n_samples(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("n_samples must be an integer")
        self.__n_samples = value

    @property
    def post_selection(self) -> PostSelectionType:
        """Describes the post-selection criteria to be applied to a state."""
        return self.__post_selection

    @post_selection.setter
    def post_selection(
        self, value: PostSelectionType | Callable | None
    ) -> None:
        self.__post_selection = process_post_selection(value)

    @property
    def min_detection(self) -> int:
        """
        Stores the minimum number of photons to be measured in an experiment,
        this excludes heralded photons.
        """
        return self.__min_detection

    @min_detection.setter
    def min_detection(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("min_detection must be an integer")
        self.__min_detection = value

    @property
    def random_seed(self) -> int | None:
        """
        Stores a random seed which is used for gathering repeatable data from
        the Sampler
        """
        return self.__random_seed

    @random_seed.setter
    def random_seed(self, value: int | None) -> None:
        self.__random_seed = process_random_seed(value)

    @property
    def sampling_mode(self) -> str:
        """
        Stores the input mode of the sampler, which controls whether N valid
        inputs or outputs are produced from the system.
        """
        return self.__sampling_mode

    @sampling_mode.setter
    def sampling_mode(self, value: str) -> None:
        if value not in ["input", "output"]:
            raise ValueError(
                "Sampling mode must be set to either input or output."
            )
        self.__sampling_mode = value

    @property
    def probability_distribution(self) -> dict:
        """
        The output probability distribution for the currently set configuration
        of the Sampler. This is re-calculated as the Sampler parameters are
        changed.
        """
        if not hasattr(self, "_Sampler__backend"):
            raise AttributeError(
                "Sampler must be run with Backend().run before the probability "
                "distribution can be viewed."
            )
        if not isinstance(self.__backend, FockBackend):
            raise TypeError(
                "Probability distribution cannot be calculated with non-Fock "
                "backend."
            )
        if self._check_parameter_updates():
            # Check circuit and input modes match
            if self.circuit.input_modes != len(self.input_state):
                raise ValueError(
                    "Mismatch in number of modes between input and circuit."
                )
            # Add heralds to the included input
            modified_state = add_heralds_to_state(
                self.input_state, self.circuit.heralds["input"]
            )
            input_state = State(modified_state)
            # Then build with source
            all_inputs = self.source._build_statistics(input_state)
            # And find probability distribution
            pdist = pdist_calc(
                self.circuit._build(), all_inputs, self.__backend
            )
            # Special case to catch an empty distribution
            if not pdist:
                pdist = {State([0] * self.circuit.n_modes): 1}
            # Assign calculated distribution to attribute
            self.__probability_distribution = pdist
            herald_modes = list(self.circuit.heralds["output"].keys())
            self.__full_to_heralded = {
                s: State(remove_heralds_from_state(s, herald_modes))
                for s in pdist
            }
            self.__calculation_values = self._gen_calculation_values()
        return self.__probability_distribution

    def _run(self, backend: FockBackend) -> SamplingResult:  # type: ignore[override]
        """
        Function to sample a state from the calculated provided distribution
        many times, producing N outputs which meet any criteria.

        Args:

            backend (FockBackend) : The target backend to run the task with.

        Returns:

            SamplingResult : A dictionary containing the different output
                states and the number of counts for each one.

        """
        self.__backend = backend

        if self.sampling_mode == "input":
            return self._sample_N_inputs(
                self.n_samples,
                self.post_selection,
                self.min_detection,
                self.random_seed,
            )
        return self._sample_N_outputs(
            self.n_samples,
            self.post_selection,
            self.min_detection,
            self.random_seed,
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
            self.__probability_distribution = {
                k: v / total_p
                for k, v in self.__probability_distribution.items()
            }
        filtered_samples = []
        # Get heralds and pre-calculate items
        heralds = self.circuit.heralds["output"]
        if heralds:
            if max(heralds.values()) > 1 and not self.detector.photon_counting:
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
                    if state not in self.__full_to_heralded:
                        self.__full_to_heralded[state] = State(
                            remove_heralds_from_state(state, herald_modes)
                        )
                    hs = self.__full_to_heralded[state]
                else:
                    hs = state
                if post_select.validate(hs) and hs.n_photons >= min_detection:
                    filtered_samples.append(hs)
        counted = dict(Counter(filtered_samples))
        return SamplingResult(counted, self.input_state)

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
        heralds = self.circuit.heralds["output"]
        if heralds:
            if max(heralds.values()) > 1 and not self.detector.photon_counting:
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
                    if s not in self.__full_to_heralded:
                        self.__full_to_heralded[s] = State(
                            remove_heralds_from_state(s, herald_modes)
                        )
                    new_s = self.__full_to_heralded[s]
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
        probs = np.array(list(pdist.values()))
        probs = probs / sum(probs)
        # Put all possible states into array
        vals = np.zeros(len(pdist), dtype=object)
        for i, k in enumerate(pdist.keys()):
            vals[i] = k
        # Generate N random samples and then process and count output states
        rng = np.random.default_rng(process_random_seed(seed))
        samples = rng.choice(vals, p=probs, size=N)
        # Count states and convert to results object
        counted = dict(Counter(samples))
        return SamplingResult(counted, self.input_state)

    def _check_parameter_updates(self) -> bool:
        """
        Determines if probabilities have already been calculated with a given
        configuration and if so will return False. If they haven't been
        calculated yet or the the parameters have changed then this will return
        True.
        """
        # Return True if not already calculated
        if not hasattr(self, "_Sampler__calculation_values"):
            return True
        # Otherwise check entries in the list are equivalent, returning False
        # if this is the case and true otherwise
        for i1, i2 in zip(
            self._gen_calculation_values(),
            self.__calculation_values,
            strict=True,
        ):
            # Treat arrays and other values differently
            if isinstance(i1, np.ndarray) and isinstance(i2, np.ndarray):
                if i1.shape != i2.shape:
                    return True
                if not (i1 == i2).all():
                    return True
            else:
                if i1 != i2:
                    return True
        return False

    def _gen_calculation_values(self) -> list:
        """
        Stores all current parameters used with the sampler in a list and
        returns this.
        """
        # Store circuit unitary and input state
        vals = [
            self.circuit.U_full,
            self.circuit.heralds,
            self.input_state,
            self.__backend.name,
        ]
        # Loop through source parameters and add these as well
        for prop in [
            "brightness",
            "purity",
            "indistinguishability",
            "probability_threshold",
        ]:
            vals.append(getattr(self.source, prop))  # noqa: PERF401
        return vals
