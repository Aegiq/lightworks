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
from typing import Any

from multimethod import multimethod

from lightworks.emulator.state import AnnotatedState
from lightworks.sdk.circuit.photonic_compiler import CompiledPhotonicCircuit
from lightworks.sdk.state import State


@multimethod
def pdist_calc(
    circuit: CompiledPhotonicCircuit,
    inputs: dict[State, int | float],
    probability_func: Callable[
        [CompiledPhotonicCircuit, State], dict[State, float]
    ],
) -> dict[State, float]:
    """
    Calculate the output state probability distribution for cases where
    inputs are state objects. This is the case when the source is perfect
    or only an imperfect brightness is used.

    Args:

        circuit (CompiledPhotonicCircuit) : The compiled circuit that is to be
            sampled from.

        inputs (dict) : The inputs to the system and their associated
            probabilities.

        probability_func (Callable) : A method for calculation of a probability
            distribution, given a circuit and input state.

    Returns:

        dict : The calculated output probability distribution.

    """
    pdist: dict[State, float] = {}
    # Loop over each possible input
    for istate, prob in inputs.items():
        # Calculate sub distribution
        sub_dist = probability_func(circuit, istate)
        if not pdist:
            if prob == 1:
                pdist = sub_dist
            else:
                pdist = {s: p * prob for s, p in sub_dist.items()}
        else:
            for s, p in sub_dist.items():
                if s in pdist:
                    pdist[s] += p * prob
                else:
                    pdist[s] = p * prob
    # Calculate zero photon state probability afterwards
    total_prob = sum(pdist.values())
    if total_prob < 1 and circuit.loss_modes > 0:
        pdist[State([0] * circuit.n_modes)] = 1 - total_prob

    return pdist


@pdist_calc.register
def annotated_state_pdist_calc(
    circuit: CompiledPhotonicCircuit,
    inputs: dict[AnnotatedState, int | float],
    probability_func: Callable[
        [CompiledPhotonicCircuit, State], dict[State, float]
    ],
) -> dict[State, float]:
    """
    Perform output state probability distribution calculation using complex
    annotated states, with imperfect purity and/or indistinguishability.

    Args:

        circuit (CompiledPhotonicCircuit) : The compiled circuit that is to be
                                    sampled from.

        inputs (dict) : The inputs to the system and their associated
                        probabilities.

        probability_func (Callable) : A method for calculation of a probability
            distribution, given a circuit and input state.

    Returns:

        dict : The calculated output probability distribution.

    """
    # Determine the input state combinations given the labels
    unique_inputs: set[State] = set()
    input_combinations: dict[AnnotatedState, list[State]] = {}
    for state in inputs:
        # Find all labels in a given state
        all_labels = []
        for mode in state:
            all_labels += mode
        # For all labels break them down into the corresponding states
        if all_labels:
            results = {lab: [0] * circuit.n_modes for lab in all_labels}
            for i, mode in enumerate(state):
                for m in mode:
                    results[m][i] += 1
            states = [State(s) for s in results.values()]
            unique_inputs |= set(states)
        else:  # Special case for empty annotated state
            states = [State([0] * circuit.n_modes)]
            unique_inputs.add(State([0] * circuit.n_modes))
        input_combinations[state] = states
    # For each of the unique inputs then need to work out the probability
    # distribution
    unique_results: dict[State, Any] = {}
    for in_state in unique_inputs:
        # Calculate sub distribution and store
        unique_results[in_state[: circuit.n_modes]] = probability_func(
            circuit, in_state
        )

    # Pre-calculate dictionary items to improve speed
    for r, pdist in unique_results.items():
        unique_results[r] = list(pdist.items())
    # Then combine the results above to work out the true output
    # probability for the inputs.
    stats_dict = {}
    for istate, combination in input_combinations.items():
        pdist = {}
        # Loop over states included in each combination
        for d in combination:
            if not pdist:
                pdist = dict(unique_results[d])
            else:
                new_pdist = {}
                # Combine existing distribution with new one
                for output, p1 in pdist.items():
                    for result, p2 in unique_results[d]:
                        new_state = output.merge(result)
                        if new_state not in new_pdist:
                            new_pdist[new_state] = p1 * p2
                        else:
                            new_pdist[new_state] += p1 * p2
                pdist = new_pdist
        # Then combine outputs and weight by input probability
        ip = inputs[istate]
        for ostate, op in pdist.items():
            if ostate not in stats_dict:
                stats_dict[ostate] = ip * op
            else:
                stats_dict[ostate] += ip * op
    return stats_dict
