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

class SamplingResult:
    """
    Stores results data from a sampling experiment in the emulator. There is 
    then a range of options for displaying the data, or alternatively the data 
    can be accessed directly using the [] operator on the class to select which
    output is required.
    
    Args:
    
        results (dict | np.ndarray) : The results which are to be stored.
        
        input (State) : The input state used in the sampling experiment.
               
    """
    
    def __init__(self, results: dict, input: State, **kwargs) -> None:
        
        if not isinstance(input, State):
            raise ResultCreationError("Input state should have type State.")
        self.__input = input
        self.__dict = results
        self.__outputs = list(results.keys())
        # Store any additional provided data from kwargs as attributes    
        for k in kwargs:
            setattr(self, k, kwargs[k])
        
        return
    
    @property
    def dictionary(self) -> dict:
        """Stores the raw results dictionary generated in the experiment."""
        return self.__dict
    
    @property
    def input(self) -> State:
        """The input state used in the sampling experiment."""
        return self.__input
    
    @property
    def outputs(self) -> State:
        """All outputs measured in the sampling experiment."""
        return self.__outputs
    
    def __getitem__(self, item: State) -> float | dict:
        """Custom get item behaviour - used when object accessed with []."""
        if isinstance(item, State):
            if item in self.dictionary:
                return self.dictionary[item]
            else:
                raise KeyError("Provided output state not in data.")
        else:
            raise ValueError("Get item value must be a State.")
    
    def __str__(self) -> str:
        return str(self.dictionary)
    
    def __len__(self) -> int:
        return len(self.dictionary)
    
    def __iter__(self) -> iter:
        """Iterable to allow to do 'for output in SamplingResult'."""
        for p in self.dictionary:
            yield p
    
    def items(self) -> iter:
        return self.dictionary.items()
    
    def keys(self) -> iter:
        return self.dictionary.keys()
    
    def values(self) -> iter:
        return self.dictionary.values()
        
    def apply_threshold_mapping(self, invert: bool = False
                                ) -> "SamplingResult":
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
        mapped_result = {}
        for out_state, val in self.dictionary.items():
            new_s = State([1 if s>=1 else 0 for s in out_state])
            if invert:
                new_s = State([1-s for s in new_s])
            if new_s in mapped_result:
                mapped_result[new_s] += val
            else:
                mapped_result[new_s] = val
        return self._recombine_mapped_result(mapped_result)
    
    def apply_parity_mapping(self, invert: bool = False) -> "SamplingResult":
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
        mapped_result = {}
        for out_state, val in self.dictionary.items():
            if invert:
                new_s = State([1-(s%2) for s in out_state])
            else:
                new_s = State([s%2 for s in out_state])
            if new_s in mapped_result:
                mapped_result[new_s] += val
            else:
                mapped_result[new_s] = val
        return self._recombine_mapped_result(mapped_result)
        
    def _recombine_mapped_result(self, mapped_result: dict):
        """Creates a new Result object from mapped data."""
        return SamplingResult(mapped_result, self.input)
        
    def plot(self, show: bool = False, 
             state_labels: dict = {}) -> tuple | None:
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
        # Check provided state labels are valid
        for state, label in state_labels.items():
            if not isinstance(state, State):
                raise TypeError("Keys of state_labels dict should be States.")
            # Convert values from state_labels to strings if not already
            state_labels[state] = str(label)
        
        fig, ax = plt.subplots(figsize = (7,6))
        x_data = range(len(self.dictionary))
        ax.bar(x_data, self.dictionary.values())
        ax.set_xticks(x_data)
        labels = [state_labels[s] if s in state_labels else str(s) 
                  for s in self.dictionary]
        ax.set_xticklabels(labels, rotation = 90)
        ax.set_xlabel("State")
        ax.set_ylabel("Counts")
           
        # Optionally use show on plot if specified
        if show:
            plt.show()
            return 
        else:
            return (fig, ax)
    
    def print_outputs(self) -> None:
        """
        Print the output results for each input into the system. This is 
        compatible with all possible result types.
        """

        to_print = str(self.input) + " -> "
        for ostate, p in self.dictionary.items():
            to_print += str(ostate) + " : " + str(p) + ", "
        to_print = to_print[:-2]    
        print(to_print)
                
        return
   
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
        data = np.array(list(self.dictionary.values()))
        # Apply thresholding to values
        for i in range(data.shape[0]):
            val = data[i]
            re = np.real(val) if abs(np.real(val)) > threshold else 0
            im = np.imag(val) if abs(np.imag(val)) > threshold else 0
            data[i] = re if abs(im) == 0 else re + 1j*im
        # Convert array to floats when not non complex results used
        data = data.astype(int)
        # Create dataframe
        df = pd.DataFrame(data, index = out_strings, columns = in_strings)
        return df.transpose()