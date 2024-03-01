"""
Class to simulate detector response when sampling, including detector 
efficiency, dark counts and non-photon number resolving detectors.
"""

from ...sdk import State

from random import random
from numbers import Number

class Detector:
    """
    Detector class
    This class can be used to sample expected output states for a given input
    and detector response.
    
    Args:
    
        efficiency (float, optional) : Set the per-channel efficiency of the 
            detectors.
        
        p_dark (float, optional) : The probability of dark counts per detector,
            note that this will depend on the dark count rate and the system 
            clock speed.
        
        photon_counting (bool, optional) : Set to True to use photon number
            resolving detectors and False to use threshold detection.
                                           
        preset (list | None, optional) : Choose from a set of preset detectors,
            this should be provided as a list where the first value is a string
            containing the detector type, and the second is the system clock 
            speed in hz for calculation of the dark count probability. For 
            example, a valid argument would be ["APD", 1e6]. Note that this 
            will overwrite any other provided settings if set to anything other
            than None.
            
    """
    def __init__(self, efficiency: float = 1, p_dark: float = 0, 
                 photon_counting: bool = True, 
                 preset: str | None = None) -> None:
        if preset is None:
            # Assign to attributes
            self.efficiency = efficiency
            self.p_dark = p_dark
            self.photon_counting = photon_counting
        else:
            det, clock = preset
            # Store presets in a dictionary, with dictionaries of values
            detectors = {"APD" : {"efficiency" : 0.5,
                                  "dark_count_rate" : 5000,
                                  "photon_counting" : False},
                         "SNSPD" : {"efficiency" : 0.85,
                                    "dark_count_rate" : 10,
                                    "photon_counting" : False},
                         "PNR" : {"efficiency" : 0.95,
                                  "dark_count_rate" : 10,
                                  "photon_counting" : True},
                         }
            # Check preset in programmed values
            if det not in detectors:
                msg = "Detector type not in presets, should be in list: ["
                for d in detectors:
                    msg = msg + d + ", "
                msg = msg[:-2] + "]."
                raise KeyError(msg)
            # If found then assign attributes
            self.efficiency = detectors[det]["efficiency"]
            self.p_dark = detectors[det]["dark_count_rate"]/clock
            self.photon_counting = detectors[det]["photon_counting"]
        return
    
    @property
    def efficiency(self) -> float:
        """The per-channel detection efficiency."""
        return self.__efficiency
    
    @efficiency.setter
    def efficiency(self, value: float) -> None:
        if not isinstance(value, Number) or isinstance(value, bool):
            raise TypeError("efficiency value should be numeric.")
        if not 0 <= value <= 1:
            raise ValueError("Value of efficiency should be in range [0,1].")
        self.__efficiency = value
        
    @property
    def p_dark(self) -> float:
        """The per-channel dark counts probability."""
        return self.__p_dark
    
    @p_dark.setter
    def p_dark(self, value: float) -> None:
        if not isinstance(value, Number) or isinstance(value, bool):
            raise TypeError("p_dark value should be numeric.")
        if not 0 <= value <= 1:
            raise ValueError("Value of p_dark should be in range [0,1].")
        self.__p_dark = value
        
    @property
    def photon_counting(self) -> float:
        """Controls whether the detectors are photon number resolving."""
        return self.__photon_counting
    
    @photon_counting.setter
    def photon_counting(self, value: float) -> None:
        if not isinstance(value, bool):
            raise TypeError("photon_counting should be a boolean.")
        self.__photon_counting = value
            
    
    def _get_output(self, in_state: State) -> State:
        """
        Sample an output state from the provided input.
        
        Args:
        
            in_state (State) : The input state to the detection module.
            
        Returns:
        
            State: The processed output state.
            
        """
        # Convert state to list
        output = [i for i in in_state]
        # Account for efficiency
        if self.efficiency < 1:
            for mode, n in enumerate(in_state):
                for i in range(n):
                    if random() > self.efficiency:
                        output[mode] -= 1
        # Then include dark counts
        if self.p_dark > 0:
            for mode in range(len(in_state)):
                if random() < self.p_dark:
                    output[mode] += 1
        # Also account for non-photon counting detectors
        if not self.photon_counting:
            output = [1 if count >= 1 else 0 for count in output]
        
        return State(output)  