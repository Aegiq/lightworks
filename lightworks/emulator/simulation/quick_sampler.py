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
from random import random
from typing import Callable

import numpy as np

from ...sdk.circuit import Circuit
from ...sdk.state import State
from ...sdk.utils import add_heralds_to_state, process_random_seed
from ...sdk.utils.post_selection import PostSelectionType
from ..backend import Backend
from ..results import SamplingResult
from ..utils import (
    EmulatorError,
    ModeMismatchError,
    fock_basis,
    process_post_selection,
)


class QuickSampler:
    """
    Randomly samples from the photon number distribution output of a provided
    circuit. It is designed to provide quicker sampling in cases where a
    certain set of assumptions can be made.
    These assumptions are:
    1) Photon number is preserved between the input and output.
    2) The source and detectors are perfect, with the exception of the ability
    to include non photon number resolving detectors.

    Args:

        circuit : The circuit to sample output states from.

        input_state (State) : The input state to use with the circuit for
            sampling.

        photon_counting (bool, optional) : Toggle whether or not photon number
            resolving detectors are used.

        post_select (PostSelection | function, optional) : A PostSelection
            object or function which applies a provided set of post-selection
            criteria to a state.

    """

    def __init__(
        self,
        circuit: Circuit,
        input_state: State,
        photon_counting: bool = True,
        post_select: PostSelectionType | Callable | None = None,
    ) -> None:
        # Assign parameters to attributes
        self.circuit = circuit
        self.input_state = input_state
        self.post_select = post_select  # type: ignore
        self.photon_counting = photon_counting
        self.__backend = Backend("permanent")

        return

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
            raise ModeMismatchError("Incorrect input length.")
        # Also validate state values
        value._validate()
        self.__input_state = value

    @property
    def post_select(self) -> PostSelectionType:
        """A function to be used for post-selection of outputs."""
        return self.__post_select

    @post_select.setter
    def post_select(self, value: PostSelectionType | Callable | None) -> None:
        self.__post_select = process_post_selection(value)

    @property
    def photon_counting(self) -> bool:
        """Details whether photon number resolving detectors are in use."""
        return self.__photon_counting

    @photon_counting.setter
    def photon_counting(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("Photon counting should be set to a boolean.")
        self.__photon_counting = bool(value)

    @property
    def probability_distribution(self) -> dict[State, float]:
        """
        The output probability distribution for the currently set configuration
        of the QuickSampler. This is re-calculated as the QuickSampler
        parameters are changed.
        """
        if self._check_parameter_updates():
            # Check circuit and input modes match
            if self.circuit.input_modes != len(self.input_state):
                raise ValueError(
                    "Mismatch in number of modes between input and circuit."
                )
            # For given input work out all possible outputs
            out_states = fock_basis(
                len(self.input_state), self.input_state.n_photons
            )
            if not self.photon_counting:
                out_states = [s for s in out_states if max(s) == 1]
            out_states = [s for s in out_states if self.post_select.validate(s)]
            if not out_states:
                raise ValueError(
                    "Heralding function removed all possible outputs."
                )
            # Find the probability distribution
            pdist = self._calculate_probabiltiies(out_states)
            # Check some states are found
            if not pdist:
                raise EmulatorError(
                    "No valid outputs found with the provided QuickSampler "
                    "configuration."
                )
            # Then assign calculated distribution to attribute
            self.__probability_distribution = pdist
            self.__calculation_values = self._gen_calculation_values()
            # Also pre-calculate continuous distribution
            self.__continuous_distribution = self._convert_to_continuous(pdist)
        return self.__probability_distribution

    @property
    def continuous_distribution(self) -> dict:
        """
        The probability distribution as a continuous distribution. This can be
        used for random sampling from the distribution.
        """
        return self.__continuous_distribution

    def sample(self) -> State:
        """
        Function to sample a state from the provided distribution.

        Returns:

            State : The sampled output state from the circuit.

        """
        # Get random number and find first point continuous distribution is
        # below this value
        pval = random()
        for state, cd in self.continuous_distribution.items():
            if pval < cd:
                # Return this as the found state
                return state
        return state

    def sample_N_outputs(  # noqa: N802
        self,
        N: int,  # noqa: N803
        seed: int | None = None,
    ) -> SamplingResult:
        """
        Function to sample a state from the calculated provided distribution
        many times, producing N outputs which meet any criteria.

        Args:

            N (int) : The number of samples to take from the circuit.

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
        samples = rng.choice(vals, p=list(pdist.values()), size=N)
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
        if not hasattr(self, "_QuickSampler__calculation_values"):
            return True
        # Otherwise check entries in the list are equivalent, returning False
        # if this is the case and true otherwise
        for i1, i2 in zip(
            self._gen_calculation_values(), self.__calculation_values
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
        return [
            self.__circuit.U_full,
            self.input_state,
            self.post_select,
            self.photon_counting,
        ]

    def _calculate_probabiltiies(self, outputs: list) -> dict:
        """
        Calculate the probabilities of all of the provided outputs and returns
        this as a normalised distribution.
        """
        # Build circuit to avoid unnecessary recalculation of quantities
        built_circuit = self.circuit._build()
        # Include circuit heralds to input
        in_state = add_heralds_to_state(
            self.input_state, self.circuit.heralds["input"]
        )
        # Add extra states on input for loss modes here when included
        in_state += [0] * built_circuit.loss_modes
        pdist = {}
        out_heralds = self.circuit.heralds["output"]
        # Loop through possible outputs and calculate probability of each
        for ostate in outputs:
            full_ostate = add_heralds_to_state(ostate, out_heralds)
            full_ostate += [0] * built_circuit.loss_modes
            p = self.__backend.probability(
                built_circuit.U_full, in_state, full_ostate
            )
            # Add output to distribution
            if p > 0:
                pdist[State(ostate)] = p
        # Normalise probability distribution
        p_total = sum(pdist.values())
        for s, p in pdist.items():
            pdist[s] = p / p_total
        return pdist

    def _convert_to_continuous(self, dist: dict) -> dict:
        """
        Convert a probability distribution to continuous for sampling. Note
        that this function assumes the provided distribution is already
        normalised.
        """
        cdist, pcon = {}, 0
        for s, p in dist.items():
            pcon += p
            cdist[s] = pcon
        return cdist
