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

from .probability_distribution import ProbabilityDistributionCalc as PDC
from ..utils import StateError, get_statistic_type
from ...emulator import Source, Detector
from ..results import SamplingResult
from ...sdk import State, Circuit

import numpy as np
from random import random
from types import FunctionType
from collections import Counter
from typing import Any


class Sampler:
    """
    Sampler class
    This class can be used to randomly sample from the photon number 
    distribution output of a provided circuit. The distribution is calculated
    when the class is first called and then used to return output states with
    the sample function.
    
    Args:
    
        circuit : The circuit to sample output states from.
        
        input_state (State) : The input state to use with the circuit for 
            sampling.
        
        source (Source, optional) : Provide a source object to simulate an 
            imperfect input. This defaults to None, which will create a perfect
            source.
                                    
        detector (Detector, optional) : Provide detector to simulate imperfect
            detector probabilities. This defaults to None, which will assume a
            perfect detector.
    
    """
    
    def __init__(self, circuit: Circuit, input_state: State, 
                 source: Source = None, detector: Detector = None) -> None:
        
        # Assign provided quantities to attributes
        self.circuit = circuit
        self.input_state = input_state
        self.source = source
        self.detector = detector
        
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
            msg = "Provided circuit should be a Circuit or Unitary object."
            raise TypeError(msg)
        self.__circuit = value
        
    @property
    def input_state(self) -> State:
        """The input state to be used for sampling."""
        return self.__input_state
    
    @input_state.setter
    def input_state(self, value: State) -> None:
        if type(value) != State:
            raise TypeError("A single input of type State should be provided.")
        if len(value) != self.circuit.n_modes:
            raise StateError("Incorrect input length.")
        self.__input_state = value
        
    @property
    def source(self) -> Source:
        """
        Details the properties of the Source used for creation of inputs to 
        the Sampler."""
        return self.__source
    
    @source.setter
    def source(self, value) -> None:
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
    def detector(self, value) -> None:
        if value is None:
            value = Detector()
        if not isinstance(value, Detector):
            raise TypeError("Provided detector should be a Detector object.")
        self.__detector = value
        
    @property
    def probability_distribution(self) -> dict:
        """
        The output probability distribution for the currently set configuration
        of the Sampler. This is re-calculated as the Sampler parameters are 
        changed.
        """
        if self._check_parameter_updates():
            # Check circuit and input modes match
            if self.circuit.n_modes != len(self.input_state):
                msg = "Mismatch in number of modes between input and circuit."
                raise ValueError(msg)
            # Check inputs are valid depending on the statistics 
            self._check_stats_type()
            all_inputs = self.source._build_statistics(self.input_state)
            if isinstance(next(iter(all_inputs)), State):
                pdist = PDC.state_prob_calc(self.circuit._build(), all_inputs)
            else:
                pdist = PDC.annotated_state_prob_calc(self.circuit._build(), 
                                                    all_inputs)
            # Special case to catch an empty distribution
            if not pdist:
                pdist = {State([0]*self.circuit.n_modes) : 1}
            # Assign calculated distribution to attribute
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
        if self._check_parameter_updates():
            self.probability_distribution
        return self.__continuous_distribution
        
    def sample (self) -> State:
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
                break
        # Return this as the found state - only return modes of interest
        return self.detector._get_output(state)
    
    def sample_N(self, N: int, herald: FunctionType = None, 
                 post_select: int = 0, seed: int|None = None) -> list:
        """
        Function to sample from a circuit N times, this will return the 
        measured output count distribution.
        
        Args:
        
            N (int) : The number of samples to take from the circuit.
            
            herald (function, optional) : A function which applies a provided
                set of heralding checks to a state.
                                      
            post_select (int, optional) : Post-select on a given minimum total
                number of photons, this should include any heralded photons.
                                          
            seed (int|None, optional) : Option to provide a random seed to 
                reproducibly generate samples from the function. This is 
                optional and can remain as None if this is not required.
            
        Returns:
        
            list : A list containing the measured photon number counts for each
                of the modes.
        
        """
        # Create always true herald if one isn't provided
        if herald is None:
            herald = lambda s: True
        if type(post_select) != int:
            raise TypeError("Post-selection value should be an integer.")
        if type(herald) != FunctionType:
            raise TypeError("Provided herald value should be a function.")
        n_modes = self.circuit.n_modes
        output_counts = [0]*n_modes
        pdist = self.probability_distribution
        vals = np.zeros(len(pdist), dtype=object)
        for i, k in enumerate(pdist.keys()):
            vals[i] = k
        # Generate N random samples and then process and count photons per mode
        np.random.seed(self._check_random_seed(seed))
        samples = np.random.choice(vals, p = list(pdist.values()), size = N)
        for state in samples:
            # Process state with detectors
            state = self.detector._get_output(state)
            if post_select: # Skip loop if below post select limit
                if state.num() < post_select:
                    continue
            if not herald(state): # Skip loop if it fails heralding check
                continue
            output_counts = [x + y for x, y in zip(output_counts, state)]
        return output_counts
    
    def sample_N_states(self, N: int, herald: FunctionType = None, 
                        post_select: int = 0, 
                        seed: int|None = None) -> SamplingResult:
        """
        Function to sample a state from the provided distribution many time, 
        if either heralding or post_select criteria are provided, the sampler 
        will only output states which meet these.
        
        Args:
        
            N (int) : The number of samples to take from the circuit.
        
            herald (function, optional) : A function which applies a provided
                set of heralding checks to a state.
                                      
            post_select (int, optional) : Post-select on a given minimum total
                number of photons, this should include any heralded photons.
                                          
            seed (int|None, optional) : Option to provide a random seed to 
                reproducibly generate samples from the function. This is 
                optional and can remain as None if this is not required.
        
        Returns:
        
            Result : A dictionary containing the different output states and 
                the number of counts for each one.
                    
        """
        # Create always true herald if one isn't provided
        if herald is None:
            herald = lambda s: True
        if type(post_select) != int:
            raise TypeError("Post-selection value should be an integer.")
        if type(herald) != FunctionType:
            raise TypeError("Provided herald value should be a function.")
        pdist = self.probability_distribution
        vals = np.zeros(len(pdist), dtype=object)
        for i, k in enumerate(pdist.keys()):
            vals[i] = k
        # Generate N random samples and then process and count output states
        np.random.seed(self._check_random_seed(seed))
        samples = np.random.choice(vals, p = list(pdist.values()), size = N)
        filtered_samples = []
        for state in samples:
            # Process output state
            state = self.detector._get_output(state)
            if herald(state) and state.num() >= post_select:
                filtered_samples.append(state)
        results = dict(Counter(filtered_samples))
        results = SamplingResult(results, self.input_state)
        return results
    
    def sample_N_n_photon_states(self, N: int, photon_number: int, 
                                 seed: int|None = None) -> SamplingResult:
        """
        Function to produce N samples from the probability distribution with a 
        given number of photons. This sampling method does not support detector
        dark counts.
        
        Args:
        
            N (int) : The number of samples that are to be returned.
            
            photon_number (int) : The number of photons that each sampled state
                should contain.
                                  
            seed (int|None, optional) : Option to provide a random seed to 
                reproducibly generate samples from the function. This is 
                optional and can remain as None if this is not required.
                                  
        Returns:
        
            Result : A dictionary containing the different output states and 
                the number of counts for each one.
                                         
        """
        pdist = self.probability_distribution
        # Check no detector dark counts included
        if self.detector.p_dark == 0:
            # When photon counting remove any states with incorrect photon num
            if self.detector.photon_counting:
                n_dist = {s:p for s, p in pdist.items()
                          if s.num() == photon_number}
            # Otherwise need to apply thresholding and add to new distribution
            else:
                n_dist = {}
                for s, p in pdist.items():
                    s = State([min(1,i) for i in s])
                    if s.num() == photon_number:
                        if s in n_dist:
                            n_dist[s] += p
                        else:
                            n_dist[s] = p
            # If outputs found then re-normalize distribution
            if n_dist:
                pt = sum(n_dist.values())
                n_dist = {s:p/pt for s, p in n_dist.items()}
            # Otherwise raise error
            else:
                msg = "No output states with target photon number found."
                raise ValueError(msg)
        else:
            raise ValueError("Not supported when using detector dark counts")
        # Put all possible states into array
        vals = np.zeros(len(n_dist), dtype=object)
        for i, k in enumerate(n_dist.keys()):
            vals[i] = k
        # Generate N random samples and then process and count output states
        np.random.seed(self._check_random_seed(seed))
        samples = np.random.choice(vals, p = list(n_dist.values()), size = N)
        # Count states and convert to results object
        results = dict(Counter(samples))
        results = SamplingResult(results, self.input_state)
        return results
    
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
        for i1, i2 in zip(self._gen_calculation_values(), 
                          self.__calculation_values):
            # Treat arrays and other values differently
            if isinstance(i1, np.ndarray) and isinstance(i2, np.ndarray):
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
        vals = [self.__circuit.U_full, self.input_state]
        # Loop through source parameters and add these as well
        for prop in ["brightness", "purity", "indistinguishability", 
                     "probability_threshold"]:
            vals.append(self.source.__getattribute__(prop))
        return vals
    
    def _convert_to_continuous(self, dist: dict) -> dict:
        """
        Convert a probability distribution to continuous for sampling, with 
        normalisation also being applied."""
        cdist, pcon = {}, 0
        total = sum(dist.values())
        for s, p in dist.items():
            pcon += p
            cdist[s] = pcon/total
        return cdist
    
    def _check_stats_type(self) -> str:
        """
        Returns the current stats type and also performs some checking that the
        currently set values are valid.
        """
        stats_type = get_statistic_type()
        if stats_type == "fermionic":
            self._fermionic_checks(self.input_state, self.source)
        return stats_type 
    
    def _fermionic_checks(self, in_state: State, source: Source) -> None:
        """Perform additional checks when doing fermionic sampling."""
        if max(in_state) > 1:
            msg = """Max number of photons per mode must be 1 when using 
                        fermionic statistics."""
            raise ValueError(" ".join(msg.split()))
        if source.purity < 1:
            msg = "Fermionic sampling does not support non-ideal purity."
            raise ValueError(msg)
        if source.indistinguishability < 1:
            msg = """Fermionic sampling does not support indistinguishability
                     modification."""
            raise ValueError(" ".join(msg.split()))
        
    def _check_random_seed(self, seed: Any) -> int | None:
        """Process a supplied random seed."""
        if not isinstance(seed, (int, type(None))):
            if int(seed) == seed:
                seed = int(seed)
            else:
                raise TypeError("Random seed must be an integer.")
        return seed