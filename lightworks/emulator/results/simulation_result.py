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

from ..utils import ResultCreationError
from ...sdk.state import State

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SimulationResult:
    """
    Stores results data from a given simulation in the emulator. There is then 
    a range of options for displaying the data, or alternatively the data can 
    be accessed directly using the [] operator on the class to select which 
    input and output data is required.
    
    Args:
    
        results (dict | np.ndarray) : The results which are to be stored.
        
        result_type (str) : The type of results which are being stored. This 
            should either be probability, probability_amplitude or counts.
            
        inputs (list) : A list of the inputs used for creation of the results.
        
        outputs (list): A list of the possible outputs from the results.     
        
    """
    
    def __init__(self, results: dict | np.ndarray, result_type: str, 
                 inputs: list, outputs: list, **kwargs) -> None:
        
        # Store result_type if valid
        if result_type in ["probability", "probability_amplitude"]:
            self.__result_type = result_type
        else:
            raise ResultCreationError(
                "Valid result type not provided, should either be "
                "'probability', 'probability_amplitude'.")
        
        self.__array = np.array(results)
        self.__inputs = inputs
        self.__outputs = outputs
        if len(self.__inputs) != self.__array.shape[0]:
            raise ResultCreationError(
                "Mismatch between inputs length and array size.")
        if len(self.__outputs) != self.__array.shape[1]:
            raise ResultCreationError(
                "Mismatch between outputs length and array size.")
        
        dict_results = {}
        for i, istate in enumerate(self.__inputs):
            input_results = {}
            for j, ostate in enumerate(self.__outputs):
                input_results[ostate] = self.__array[i,j]
            dict_results[istate] = input_results
        self.__dict = dict_results
            
        # Store any additional provided data from kwargs as attributes    
        for k in kwargs:
            setattr(self, k, kwargs[k])
        
        return
    
    @property
    def array(self) -> np.ndarray:
        """
        The calculated array of data, where the first dimension corresponds to
        the inputs and the second dimension to the outputs.
        """
        return self.__array
    
    @property
    def inputs(self) -> State:
        """All inputs which values were calculated for."""
        return self.__inputs
    
    @property
    def outputs(self) -> State:
        """All outputs which values were calculated for."""
        return self.__outputs
    
    @property
    def dictionary(self) -> dict:
        """Stores a dictionary of inputs and the associated output values."""
        return self.__dict
    
    @property
    def result_type(self) -> dict:
        """
        Details where the result is a probability or probability amplitude.
        """
        return self.__result_type
    
    def __getitem__(self, item: State | tuple) -> float | dict:
        """Custom get item behaviour - used when object accessed with []."""
        if isinstance(item, State):
            # When only one input is used, automatically return output value
            # from dictionary instead of dictionary
            if len(self.inputs) == 1:
                if item in self.dictionary[self.inputs[0]]:
                    return self.dictionary[self.inputs[0]][item]
                else:
                    raise KeyError("Provided output state not in data.")
            else:
                if item in self.dictionary:
                    return self.dictionary[item]
                else:
                    raise KeyError("Provided input state not in data.")
        elif isinstance(item, tuple):
            # Check only two values have been provided
            if len(item) > 2:
                raise ValueError(
                    "Get item can only contain a maximum of two values.")
            # Separate data into two states
            istate = item[0]
            ostate = item[1]
            # Check all aspects are valid
            if (not isinstance(istate, State) or 
                not isinstance(ostate, (State, type(None)))):
                raise ValueError(
                    "Get item values should have type State.")
            if istate in self.dictionary:
                sub_r = self.dictionary[istate]
            else:
                raise KeyError("Requested input state not in data.")
            # If None provided as second value then return all results for input
            if ostate is None:
                return sub_r
            # Else return requested value
            elif ostate in sub_r:
                return sub_r[ostate]
            else:
                raise KeyError("Requested output state not in data.")
        else:
            raise ValueError(
                "Get item value must be either one or two States.")
        
    def __str__(self) -> str:
        return str(self.dictionary)
    
    def __iter__(self) -> iter:
        """Iterable to allow to do 'for input in SimulationResult'."""
        for p in self.dictionary:
            yield p
    
    def apply_threshold_mapping(self, invert: bool = False
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
                "Threshold mapping cannot be applied to probability "
                "amplitudes.")
        mapped_result = {}
        for in_state in self.inputs:
            mapped_result[in_state] = {}
            for out_state, val in self.dictionary[in_state].items():
                new_s = State([1 if s>=1 else 0 for s in out_state])
                if invert:
                    new_s = State([1-s for s in new_s])
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
                "Parity mapping cannot be applied to probability amplitudes.")
        mapped_result = {}
        for in_state in self.inputs:
            mapped_result[in_state] = {}
            for out_state, val in self.dictionary[in_state].items():
                if invert:
                    new_s = State([1-(s%2) for s in out_state])
                else:
                    new_s = State([s%2 for s in out_state])
                if new_s in mapped_result[in_state]:
                    mapped_result[in_state][new_s] += val
                else:
                    mapped_result[in_state][new_s] = val
        return self._recombine_mapped_result(mapped_result)
    
    def _recombine_mapped_result(self, mapped_result: dict
                                 ) -> "SimulationResult":
        """Creates a new Result object from mapped data."""
        unique_outputs = set()
        for in_state, pdist in mapped_result.items():
            for out_state in pdist:
                unique_outputs.add(out_state)
        array = np.zeros((len(self.inputs), len(unique_outputs)))
        for i, in_state in enumerate(self.inputs):
            for j, out_state in enumerate(unique_outputs):
                if out_state in mapped_result[in_state]:
                    array[i,j] = mapped_result[in_state][out_state]
        r =  SimulationResult(array, result_type = self.result_type,
                              inputs = self.inputs, 
                              outputs = list(unique_outputs))
        return r
    
    def plot(self, conv_to_probability: bool = False, show: bool = False,
             state_labels: dict = {}) -> tuple | None:
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
        # TODO: Try to clean up the logic here
        # Check provided state labels are valid
        for state, label in state_labels.items():
            if not isinstance(state, State):
                raise TypeError("Keys of state_labels dict should be States.")
            # Convert values from state_labels to strings if not already
            state_labels[state] = str(label)
        # Use imshow for any results originally provided as an array if more 
        # than two more inputs are used.
        if len(self.inputs) >= 2: 
            if (self.result_type != "probability_amplitude" or 
                conv_to_probability):
                if self.result_type == "probability_amplitude":
                    data = abs(self.array)**2
                else:
                    data = abs(self.array)
                # Plot array and add colorbar
                fig, ax = plt.subplots()
                im = ax.imshow(data)
                im_ratio = data.shape[0]/data.shape[1]
                fig.colorbar(im, ax=ax, fraction=0.046*im_ratio, pad=0.04)
                # Label states on each axis
                ax.set_xticks(range(len(self.outputs)))
                xlabels = [state_labels[s] if s in state_labels else str(s) 
                           for s in self.outputs]
                ax.set_xticklabels(xlabels, rotation = 90)
                ax.set_yticks(range(len(self.inputs)))
                ylabels = [state_labels[s] if s in state_labels else str(s) 
                           for s in self.inputs]
                ax.set_yticklabels(ylabels)
            else:
                fig, ax = plt.subplots(1, 2, figsize = (20,6))
                vmin = min([op(self.array).min() for op in [np.real, np.imag]])
                vmax = min([op(self.array).max() for op in [np.real, np.imag]])
                im = ax[0].imshow(np.real(self.array), vmin=vmin, vmax=vmax)
                ax[0].set_title("real(result)")
                ax[1].imshow(np.imag(self.array), vmin=vmin, vmax=vmax)
                ax[1].set_title("imag(result)")
                for i in range(2):
                    ax[i].set_xticks(range(len(self.outputs)))
                    xlabels = [state_labels[s] if s in state_labels else str(s) 
                               for s in self.outputs]
                    ax[i].set_xticklabels(xlabels, rotation = 90)
                    ax[i].set_yticks(range(len(self.inputs)))
                    ylabels = [state_labels[s] if s in state_labels else str(s) 
                               for s in self.inputs]
                    ax[i].set_yticklabels(ylabels)
                fig.colorbar(im, ax=ax.ravel().tolist())
        # Otherwise use a bar chart
        else:
            istate = self.inputs[0]
            if (self.result_type != "probability_amplitude" or 
                conv_to_probability):
                if self.result_type == "probability_amplitude":
                    data = {}
                    for s, p in self.dictionary[istate].items():
                        data[s] = abs(p)**2
                    title = "Probability"
                else:
                    data = self.dictionary[istate]
                    title = self.result_type.capitalize()
                    
                fig, ax = plt.subplots(figsize = (7,6))
                x_data = range(len(data))
                ax.bar(x_data, data.values())
                ax.set_xticks(x_data)
                labels = [state_labels[s] if s in state_labels else str(s) 
                          for s in data]
                ax.set_xticklabels(labels, rotation = 90)
                ax.set_xlabel("State")
                ax.set_ylabel(title)
            # Plot both real and imaginary parts
            else:
                fig, ax = plt.subplots(1, 2, figsize = (14,6))
                x_data = range(len(self.dictionary[istate]))
                ax[0].bar(x_data, 
                          np.real(list(self.dictionary[istate].values())))
                ax[1].bar(x_data, 
                          np.imag(list(self.dictionary[istate].values())))
                for i in range(2):
                    ax[i].set_xticks(x_data)
                    labels = [state_labels[s] if s in state_labels else str(s) 
                              for s in self.dictionary[istate]]
                    ax[i].set_xticklabels(labels, rotation = 90)
                    ax[i].set_xlabel("State")
                    ax[i].axhline(0, color = "black", linewidth = 0.5)
                ax[0].set_ylabel("real(amplitude)")
                ax[1].set_ylabel("imag(amplitude)")
        # Optionally use show on plot if specified
        if show:
            plt.show()
            return 
        
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
        for istate in self.inputs:
            to_print = str(istate) + " -> "
            for ostate, p in self.dictionary[istate].items():
                # Adjust print order based on quantity
                if self.result_type == "counts":
                    to_print += str(ostate) + " : " + str(p) + ", "
                else:
                    p = np.round(p, rounding)
                    if abs(p.real) > 0 or abs(p.imag) > 0:
                        to_print += str(p) + "*" + str(ostate) + " + "
            to_print = to_print[:-2]    
            print(to_print)
                
        return
    
    def display_as_dataframe(self, threshold: float = 1e-12, 
                             conv_to_probability: bool = False
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
        if conv_to_probability:
            if self.result_type == "probability_amplitude":
                data = abs(data)**2
        # Apply thresholding to values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i,j]
                re = np.real(val) if abs(np.real(val)) > threshold else 0
                im = np.imag(val) if abs(np.imag(val)) > threshold else 0
                data[i,j] = re if abs(im) == 0 else re + 1j*im
        # Convert array to floats when not non complex results used
        if self.result_type != "probability_amplitude" or conv_to_probability:
            data = abs(data)
            if self.result_type == "counts" and not conv_to_probability:
                data = data.astype(int)
            else:
                data = data.astype(float)
        # Create dataframe
        df = pd.DataFrame(data, index = in_strings, columns = out_strings)
        return df