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

from collections.abc import Callable
from types import NoneType
from typing import Literal

from lightworks.emulator.components import Detector, Source
from lightworks.sdk.circuit import PhotonicCircuit
from lightworks.sdk.results import ProbabilityDistribution
from lightworks.sdk.state import State
from lightworks.sdk.utils.exceptions import ModeMismatchError, SamplerError
from lightworks.sdk.utils.post_selection import (
    PostSelectionType,
    process_post_selection,
)
from lightworks.sdk.utils.random import process_random_seed
from lightworks.sdk.utils.state import check_herald_difference

from .data import SamplerTask
from .task import Task


class Sampler(Task):
    """
    Used to randomly sample from the photon number distribution output of a
    provided circuit. The distribution is calculated when the class is first
    called and then used to return output states with the sample function. Also
    supports the inclusion of imperfect source and detection properties.

    Args:

        circuit (PhotonicCircuit) : The circuit to sample output states from.

        input_state (State) : The input state to use with the circuit for
            sampling.

        n_samples (int) : The number of samples that are to be returned.

        source (Source, optional) : Provide a source object to simulate an
            imperfect input. This defaults to None, which will create a perfect
            source.

        detector (Detector, optional) : Provide detector to simulate imperfect
            detector probabilities. This defaults to None, which will assume a
            perfect detector.

        post_selection (PostSelection | function, optional) : A PostSelection
            object or function which applies a provided set of post-selection
            criteria to a state.

        min_detection (int | None, optional) : Post-select on a given minimum
            total number of photons, this should not include any heralded
            photons. If set to None then this will be the smallest valuable
            possible on a platform.

        random_seed (int|None, optional) : Option to provide a random seed to
            reproducibly generate samples from the function. This is
            optional and can remain as None if this is not required.

        sampling_mode (str, optional) : Sets the mode of the Sampler. In input
            mode, N cycles of the system are measured, and only results which
            meet any assigned criteria are returned. In output mode, N valid
            samples are produced from the system. Should be either 'input' or
            'output', defaults to 'output'.

    """

    def __init__(
        self,
        circuit: PhotonicCircuit,
        input_state: State,
        n_samples: int,
        post_selection: PostSelectionType
        | Callable[[State], bool]
        | None = None,
        min_detection: int | None = None,
        source: Source | None = None,
        detector: Detector | None = None,
        random_seed: int | None = None,
        sampling_mode: Literal["input", "output"] = "output",
    ) -> None:
        # Assign provided quantities to attributes
        self.circuit = circuit
        self.input_state = input_state
        self.source = source
        self.detector = detector
        self.n_samples = n_samples
        self.post_selection = post_selection  # type: ignore[assignment]
        self.min_detection = min_detection
        self.random_seed = random_seed
        self.sampling_mode = sampling_mode
        self._probability_distribution: ProbabilityDistribution

    @property
    def circuit(self) -> PhotonicCircuit:
        """
        Stores the circuit to be used for simulation, should be a
        PhotonicCircuit object.
        """
        return self.__circuit

    @circuit.setter
    def circuit(self, value: PhotonicCircuit) -> None:
        if not isinstance(value, PhotonicCircuit):
            raise TypeError(
                "Provided circuit should be a PhotonicCircuit or Unitary "
                "object."
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
    def source(self) -> Source | None:
        """
        Details the properties of the Source used for creation of inputs to
        the Sampler.
        """
        return self.__source

    @source.setter
    def source(self, value: Source | None) -> None:
        if not isinstance(value, Source | NoneType):
            raise TypeError("Provided source should be a Source object.")
        self.__source = value

    @property
    def detector(self) -> Detector | None:
        """
        Details the properties of the Detector used for photon measurement.
        """
        return self.__detector

    @detector.setter
    def detector(self, value: Detector | None) -> None:
        if not isinstance(value, Detector | NoneType):
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
        if value < 0:
            raise ValueError("Number of samples should be larger than 0.")
        self.__n_samples = value

    @property
    def post_selection(self) -> PostSelectionType | None:
        """Describes the post-selection criteria to be applied to a state."""
        return self.__post_selection

    @post_selection.setter
    def post_selection(
        self, value: PostSelectionType | Callable[[State], bool] | None
    ) -> None:
        self.__post_selection = process_post_selection(value)

    @property
    def min_detection(self) -> int | None:
        """
        Stores the minimum number of photons to be measured in an experiment,
        this excludes heralded photons.
        """
        return self.__min_detection

    @min_detection.setter
    def min_detection(self, value: int | None) -> None:
        if not isinstance(value, int | NoneType) or isinstance(value, bool):
            raise TypeError("min_detection must be an integer or None.")
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
    def sampling_mode(self) -> Literal["input", "output"]:
        """
        Stores the input mode of the sampler, which controls whether N valid
        inputs or outputs are produced from the system.
        """
        return self.__sampling_mode

    @sampling_mode.setter
    def sampling_mode(self, value: Literal["input", "output"]) -> None:
        if value not in {"input", "output"}:
            raise ValueError(
                "Sampling mode must be set to either input or output."
            )
        self.__sampling_mode = value

    @property
    def probability_distribution(self) -> ProbabilityDistribution:
        """
        Returns the calculated probability distribution. This can only be called
        once the Sampler has been run on the backend. Note: If any changes are
        made to the Sampler configuration, it should be re-evaluated with the
        backend to update this distribution.
        """
        if not hasattr(self, "_probability_distribution"):
            raise SamplerError(
                "Probability distribution must be calculated by first running "
                "the Sampler on a backend."
            )
        return self._probability_distribution

    def _generate_task(self) -> SamplerTask:
        check_herald_difference(self.circuit, self.input_state.n_photons)
        return SamplerTask(
            circuit=self.circuit._build(),
            input_state=self.input_state,
            n_samples=self.n_samples,
            source=self.source,
            detector=self.detector,
            post_selection=self.post_selection,
            min_detection=self.min_detection,
            random_seed=self.random_seed,
            sampling_mode=self.sampling_mode,
        )
