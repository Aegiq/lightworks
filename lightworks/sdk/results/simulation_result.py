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

from typing import Any, overload

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..state import State
from ..utils import ResultCreationError


class SimulationResult(dict[State, dict[State, float | complex]]):
    """
    Stores results data from a given simulation in the emulator. There is then
    a range of options for displaying the data, or alternatively the data can
    be accessed directly using the [] operator on the class to select which
    input and output data is required.

    Args:

        results (np.ndarray) : The results which are to be stored.

        result_type (str) : The type of results which are being stored. This
            should either be probability, probability_amplitude or counts.

        inputs (list) : A list of the inputs used for creation of the results.

        outputs (list): A list of the possible outputs from the results.

    """

    def __init__(
        self,
        results: NDArray[np.float64 | np.complex128],
        result_type: str,
        inputs: list[State],
        outputs: list[State],
        **kwargs: Any,
    ) -> None:
        # Store result_type if valid
        if result_type in ["probability", "probability_amplitude"]:
            self.__result_type = result_type
        else:
            raise ResultCreationError(
                "Valid result type not provided, should either be "
                "'probability', 'probability_amplitude'."
            )

        self.__array = np.array(results)
        self.__inputs = inputs
        self.__outputs = outputs
        if len(self.__inputs) != self.__array.shape[0]:
            raise ResultCreationError(
                "Mismatch between inputs length and array size."
            )
        if len(self.__outputs) != self.__array.shape[1]:
            raise ResultCreationError(
                "Mismatch between outputs length and array size."
            )

        dict_results = {}
        for i, istate in enumerate(self.__inputs):
            input_results = {}
            for j, ostate in enumerate(self.__outputs):
                input_results[ostate] = self.__array[i, j]
            dict_results[istate] = input_results
        super().__init__(dict_results)

        # Store any additional provided data from kwargs as attributes
        for k in kwargs:
            setattr(self, k, kwargs[k])

        return

    @property
    def array(self) -> NDArray[np.float64 | np.complex128]:
        """
        The calculated array of data, where the first dimension corresponds to
        the inputs and the second dimension to the outputs.
        """
        return self.__array

    @property
    def inputs(self) -> list[State]:
        """All inputs which values were calculated for."""
        return self.__inputs

    @property
    def outputs(self) -> list[State]:
        """All outputs which values were calculated for."""
        return self.__outputs

    @property
    def result_type(self) -> str:
        """
        Details where the result is a probability or probability amplitude.
        """
        return self.__result_type

    @overload
    def __getitem__(self, item: State) -> dict[State, float | complex]: ...

    @overload
    def __getitem__(
        self, item: tuple[State]
    ) -> dict[State, float | complex]: ...

    @overload
    def __getitem__(self, item: tuple[State, State]) -> float | complex: ...

    def __getitem__(
        self, item: State | tuple[State] | tuple[State, State]
    ) -> float | complex | dict[State, float | complex]:
        """Custom get item behaviour - used when object accessed with []."""
        if isinstance(item, State):
            if item not in self:
                raise KeyError("Requested input state not in data.")
            return super().__getitem__(item)
        if isinstance(item, tuple):
            # Check only two values have been provided
            if len(item) > 2:
                raise ValueError(
                    "Get item can only contain a maximum of two values."
                )
            # Separate data into two states
            istate = item[0]
            ostate = item[1] if len(item) == 2 else None
            # Check all aspects are valid
            if not isinstance(istate, State) or not isinstance(
                ostate, State | type(None)
            ):
                raise TypeError("Get item values should have type State.")
            sub_r = self[istate]
            # If None provided as second value then return all results for input
            if ostate is None:
                return sub_r
            # Else return requested value
            if ostate not in sub_r:
                raise KeyError("Requested output state not in data.")
            return sub_r[ostate]
        raise TypeError("Get item value must be either one or two States.")

    def apply_threshold_mapping(
        self, invert: bool = False
    ) -> "SimulationResult":
        """
        Apply a threshold mapping to the results from the object and return
        this as a dictionary.

        Args:

            invert (bool, optional) : Select whether to invert the threshold
                mapping. This will swap the 0s and 1s of the produced states.

        Returns:

            Result : A new Result containing the threshold mapped state
                distribution.

        """
        if self.result_type == "probability_amplitude":
            raise ValueError(
                "Threshold mapping cannot be applied to probability amplitudes."
            )
        mapped_result: dict[State, dict[State, float | complex]] = {}
        for in_state, results in self.items():
            mapped_result[in_state] = {}
            for out_state, val in results.items():
                new_s = State([1 if s >= 1 else 0 for s in out_state])
                if invert:
                    new_s = State([1 - s for s in new_s])
                if new_s in mapped_result[in_state]:
                    mapped_result[in_state][new_s] += val
                else:
                    mapped_result[in_state][new_s] = val
        return self._recombine_mapped_result(mapped_result)

    def apply_parity_mapping(self, invert: bool = False) -> "SimulationResult":
        """
        Apply a parity mapping to the results from the object and return this
        as a dictionary.

        Args:

            invert (bool, optional) : Select whether to invert the parity
                mapping. This will swap between even->0 & odd->1 and even->1 &
                odd->0.

        Returns:

            Result : A new Result containing the parity mapped state
                distribution.

        """
        if self.result_type == "probability_amplitude":
            raise ValueError(
                "Parity mapping cannot be applied to probability amplitudes."
            )
        mapped_result: dict[State, dict[State, float | complex]] = {}
        for in_state, results in self.items():
            mapped_result[in_state] = {}
            for out_state, val in results.items():
                if invert:
                    new_s = State([1 - (s % 2) for s in out_state])
                else:
                    new_s = State([s % 2 for s in out_state])
                if new_s in mapped_result[in_state]:
                    mapped_result[in_state][new_s] += val
                else:
                    mapped_result[in_state][new_s] = val
        return self._recombine_mapped_result(mapped_result)

    def _recombine_mapped_result(
        self, mapped_result: dict[State, dict[State, float | complex]]
    ) -> "SimulationResult":
        """Creates a new Result object from mapped data."""
        unique_outputs = set()
        for pdist in mapped_result.values():
            for out_state in pdist:
                unique_outputs.add(out_state)
        array = np.zeros((len(self.inputs), len(unique_outputs)))
        for i, in_state in enumerate(self.inputs):
            for j, out_state in enumerate(unique_outputs):
                if out_state in mapped_result[in_state]:
                    array[i, j] = mapped_result[in_state][out_state]
        return SimulationResult(
            array,
            result_type=self.result_type,
            inputs=self.inputs,
            outputs=list(unique_outputs),
        )

    def plot(
        self,
        conv_to_probability: bool = False,
        show: bool = True,
        state_labels: dict[State, str | State] | None = None,
    ) -> tuple[matplotlib.figure.Figure, plt.Axes | NDArray[Any]] | None:
        """
        Create a plot of the data contained in the result. This will either
        take the form of a heatmap or bar chart, depending on the nature of the
        data contained in the Result object.

        Args:

            conv_to_probability (bool, optional) : In the case that the result
                is a probability amplitude, setting this to True will convert
                it into a probability.

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
        # Use imshow for any results originally provided as an array if more
        # than two more inputs are used.
        if len(self.inputs) >= 2:
            fig, ax = self._plot_array(conv_to_probability, state_labels)
        # Otherwise use a bar chart
        else:
            fig, ax = self._plot_bar(conv_to_probability, state_labels)
        # Optionally use show on plot if specified
        if show:
            plt.show()
            return None
        return (fig, ax)

    def print_outputs(self, rounding: int = 4) -> None:
        """
        Print the output results for each input into the system. This is
        compatible with all possible result types.

        Args:

            rounding (int, optional) : Set the number of decimal places which
                each number will be rounded to, defaults to 4.

        """
        # Loop over each input and print results
        for istate, results in self.items():
            to_print = str(istate) + " -> "
            for ostate, p in results.items():
                # Adjust print order based on quantity
                if self.result_type == "counts":
                    to_print += str(ostate) + " : " + str(p) + ", "
                else:
                    p = np.round(p, rounding)  # noqa: PLW2901
                    if abs(p.real) > 0 or abs(p.imag) > 0:
                        to_print += str(p) + "*" + str(ostate) + " + "
            to_print = to_print[:-2]
            print(to_print)  # noqa: T201

        return

    def display_as_dataframe(
        self, threshold: float = 1e-12, conv_to_probability: bool = False
    ) -> pd.DataFrame:
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
        in_strings = [str(s) for s in self.inputs]
        out_strings = [str(s) for s in self.outputs]
        # Switch to probability if required
        data = self.array.copy()
        if conv_to_probability and self.result_type == "probability_amplitude":
            data = abs(data) ** 2  # type: ignore[assignment]
        # Apply thresholding to values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                re = np.real(val) if abs(np.real(val)) > threshold else 0
                im = np.imag(val) if abs(np.imag(val)) > threshold else 0
                data[i, j] = re if abs(im) == 0 else re + 1j * im
        # Convert array to floats when not non complex results used
        if self.result_type != "probability_amplitude" or conv_to_probability:
            data = abs(data)
            if self.result_type == "counts" and not conv_to_probability:
                data = data.astype(int)
            else:
                data = data.astype(float)
        # Create dataframe and return
        return pd.DataFrame(data, index=in_strings, columns=out_strings)

    def _plot_array(
        self, conv_to_probability: bool, state_labels: dict[State, str | State]
    ) -> tuple[matplotlib.figure.Figure, plt.Axes | NDArray[Any]]:
        """
        Plots an array of the data contained within the Result object.
        """
        # Generate x and y labels
        xlabels = [
            state_labels[s] if s in state_labels else str(s)
            for s in self.outputs
        ]
        ylabels = [
            state_labels[s] if s in state_labels else str(s)
            for s in self.inputs
        ]
        # Single plots
        if self.result_type != "probability_amplitude" or conv_to_probability:
            if self.result_type == "probability_amplitude":
                a_data = abs(self.array) ** 2
            else:
                a_data = abs(self.array)
            # Plot array and add colorbar
            fig, ax = plt.subplots()
            im = ax.imshow(a_data)
            im_ratio = a_data.shape[0] / a_data.shape[1]
            fig.colorbar(im, ax=ax, fraction=0.046 * im_ratio, pad=0.04)
            # Label states on each axis
            ax.set_xticks(range(len(self.outputs)))
            ax.set_xticklabels(xlabels, rotation=90)  # type: ignore[arg-type]
            ax.set_yticks(range(len(self.inputs)))
            ax.set_yticklabels(ylabels)  # type: ignore[arg-type]
            return fig, ax
        # Otherwise create two plots
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        axes = np.array(axes)
        vmin = min(op(self.array).min() for op in [np.real, np.imag])
        vmax = min(op(self.array).max() for op in [np.real, np.imag])
        im = axes[0].imshow(np.real(self.array), vmin=vmin, vmax=vmax)
        axes[0].set_title("real(result)")
        axes[1].imshow(np.imag(self.array), vmin=vmin, vmax=vmax)
        axes[1].set_title("imag(result)")
        for i in range(2):
            axes[i].set_xticks(range(len(self.outputs)))
            axes[i].set_xticklabels(xlabels, rotation=90)
            axes[i].set_yticks(range(len(self.inputs)))
            axes[i].set_yticklabels(ylabels)
        fig.colorbar(im, ax=axes.ravel().tolist())
        return (fig, axes)

    def _plot_bar(
        self, conv_to_probability: bool, state_labels: dict[State, str | State]
    ) -> tuple[matplotlib.figure.Figure, plt.Axes | NDArray[Any]]:
        """
        Plots bar chart with data contained in Results object. This should only
        be used if there is a single input.
        """
        istate = self.inputs[0]
        results = dict(self[istate])  # type: ignore[arg-type]
        # Vary plot depending on result type
        if self.result_type != "probability_amplitude" or conv_to_probability:
            d_data: dict[State, float]
            if self.result_type == "probability_amplitude":
                d_data = {}
                for s, p in results.items():
                    d_data[s] = abs(p) ** 2
                title = "Probability"
            else:
                d_data = results  # type: ignore[assignment]
                title = self.result_type.capitalize()

            fig, ax = plt.subplots(figsize=(7, 6))
            x_data = range(len(d_data))
            ax.bar(x_data, list(d_data.values()))
            ax.set_xticks(x_data)
            labels = [
                state_labels[s] if s in state_labels else str(s) for s in d_data
            ]
            ax.set_xticklabels(labels, rotation=90)  # type: ignore[arg-type]
            ax.set_xlabel("State")
            ax.set_ylabel(title)
            return (fig, ax)
        # Plot both real and imaginary parts
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = np.array(axes)
        x_data = range(len(results))
        axes[0].bar(x_data, np.real(list(results.values())))
        axes[1].bar(x_data, np.imag(list(results.values())))
        for i in range(2):
            axes[i].set_xticks(x_data)
            labels = [
                state_labels[s] if s in state_labels else str(s)
                for s in results
            ]
            axes[i].set_xticklabels(labels, rotation=90)
            axes[i].set_xlabel("State")
            axes[i].axhline(0, color="black", linewidth=0.5)
        axes[0].set_ylabel("real(amplitude)")
        axes[1].set_ylabel("imag(amplitude)")
        return (fig, axes)
