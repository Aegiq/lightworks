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
from types import FunctionType, MethodType
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lightworks.sdk.state import State
from lightworks.sdk.utils import PostSelectionFunction
from lightworks.sdk.utils.exceptions import ResultCreationError
from lightworks.sdk.utils.post_selection import PostSelectionType

from .result import Result


class SamplingResult(Result[State, int]):
    """
    Stores results data from a sampling experiment in the emulator. There is
    then a range of options for displaying the data, or alternatively the data
    can be accessed directly using the [] operator on the class to select which
    output is required.

    Args:

        results (dict | np.ndarray) : The results which are to be stored.

        input (State) : The input state used in the sampling experiment.

    """

    def __init__(
        self, results: dict[State, int], input: State, **kwargs: Any
    ) -> None:
        super().__init__(results)
        if not isinstance(input, State):
            raise ResultCreationError("Input state should have type State.")
        self.__input = input
        self.__outputs = list(results.keys())
        # Store any additional provided data from kwargs as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def input(self) -> State:
        """The input state used in the sampling experiment."""
        return self.__input

    @property
    def outputs(self) -> list[State]:
        """All outputs measured in the sampling experiment."""
        return self.__outputs

    def __getitem__(self, item: State) -> int:
        """Custom get item behaviour - used when object accessed with []."""
        if not isinstance(item, State):
            raise TypeError("Get item value must be a State.")
        if item not in self:
            raise KeyError("Provided output state not in data.")
        return super().__getitem__(item)

    def map(
        self,
        mapping: Callable[[State, Any], State],
        *args: Any,
        **kwargs: Any,
    ) -> "SamplingResult":
        """
        Performs a generic remapping of states based on a provided function.
        """
        if not isinstance(mapping, FunctionType | MethodType):
            raise TypeError(
                "Provided mapping should be a callable function which accepts "
                "and returns a State object."
            )
        mapped_result: dict[State, int] = {}
        for out_state, val in self.items():
            new_s = mapping(out_state, *args, **kwargs)
            if new_s in mapped_result:
                mapped_result[new_s] += val
            else:
                mapped_result[new_s] = val
        return self._recombine_mapped_result(mapped_result)

    def apply_post_selection(
        self, post_selection: PostSelectionType | Callable[[State], bool]
    ) -> "SamplingResult":
        """
        Applies an additional post-selection criteria to the stored result and
        returns this as a new object.
        """
        if isinstance(post_selection, FunctionType | MethodType):
            post_selection = PostSelectionFunction(post_selection)
        if not isinstance(post_selection, PostSelectionType):
            raise TypeError(
                "Provided post_selection should either be a PostSelection "
                "object or a callable function which accepts a state and "
                "returns a boolean to indicate whether the state is valid."
            )
        return self._recombine_mapped_result(
            {
                state: val
                for state, val in self.items()
                if post_selection.validate(state)
            }
        )

    def _recombine_mapped_result(
        self, mapped_result: dict[State, int]
    ) -> "SamplingResult":
        """Creates a new Result object from mapped data."""
        return SamplingResult(mapped_result, self.input)

    def plot(
        self,
        show: bool = True,
        state_labels: dict[State, str | State] | None = None,
    ) -> tuple[matplotlib.figure.Figure, plt.Axes] | None:
        """
        Create a plot of the data contained in the result. This will either
        take the form of a heatmap or bar chart, depending on the nature of the
        data contained in the Result object.

        Args:

            show (bool, optional) : Can be used to automatically show the
                created plot with show instead of returning the figure and
                axes.

            state_labels (dict, optional) : Provided a dictionary which can be
                used to specify alternate labels for each of the states when
                plotting. The keys of this dictionary should be States and the
                values should be strings or States.

        """
        if state_labels is None:
            state_labels = {}
        # Check provided state labels are valid
        for state, label in state_labels.items():
            if not isinstance(state, State):
                raise TypeError("Keys of state_labels dict should be States.")
            # Convert values from state_labels to strings if not already
            state_labels[state] = str(label)

        fig, ax = plt.subplots(figsize=(7, 6))
        x_data = range(len(self))
        ax.bar(x_data, list(self.values()))
        ax.set_xticks(x_data)
        labels = [
            state_labels[s] if s in state_labels else str(s) for s in self
        ]
        ax.set_xticklabels(labels, rotation=90)  # type: ignore[arg-type]
        ax.set_xlabel("State")
        ax.set_ylabel("Counts")

        # Optionally use show on plot if specified
        if show:
            plt.show()
            return None
        return (fig, ax)

    def print_outputs(self) -> None:
        """
        Print the output results for each input into the system. This is
        compatible with all possible result types.
        """
        to_print = str(self.input) + " -> "
        for ostate, p in self.items():
            to_print += str(ostate) + " : " + str(p) + ", "
        to_print = to_print[:-2]
        print(to_print)  # noqa: T201

    def display_as_dataframe(self, threshold: float = 1e-12) -> pd.DataFrame:
        """
        Function to display the results of a given simulation in a dataframe
        format. Either the probability amplitudes of the state, or the actual
        probabilities can be displayed.

        Args:

            threshold (float, optional) : Threshold to control at which point
                value are rounded to zero. If looking for very small amplitudes
                this may need to be lowered.

            conv_to_probability (bool, optional) : In the case that the result
                is a probability amplitude, setting this to True will convert
                it into a probability. If it is not a probability amplitude
                then this setting will have no effect.

        Returns:

            pd.Dataframe : A dataframe with the results and input and output
                states as labels.

        """
        # Convert state vectors into strings
        in_strings = [str(self.input)]
        out_strings = [str(s) for s in self.outputs]
        # Switch to probability if required
        data = np.array(list(self.values()))
        # Apply thresholding to values
        for i in range(data.shape[0]):
            val = data[i]
            re = np.real(val) if abs(np.real(val)) > threshold else 0
            im = np.imag(val) if abs(np.imag(val)) > threshold else 0
            data[i] = re if abs(im) == 0 else re + 1j * im
        # Convert array to floats when not non complex results used
        data = data.astype(int)
        # Create dataframe
        results = pd.DataFrame(data, index=out_strings, columns=in_strings)
        return results.transpose()
