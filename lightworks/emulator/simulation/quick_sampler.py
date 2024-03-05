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

from .backend import Backend
from ..utils import fock_basis, StateError, get_statistic_type
from ..results import SamplingResult
from ...sdk import State, Circuit
from ...sdk.circuit.circuit_compiler import CompiledCircuit

import numpy as np
from random import random
from types import FunctionType
from collections import Counter
from typing import Any

    
class QuickSampler:
    """
    QuickSampler class
    This class can be used to randomly sample from the photon number 
    distribution output of a provided circuit. It is designed to provide 
    quicker sampling in cases where a certain set of assumptions can be made. 
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
        
        herald (function, optional) : A function which applies a provided set 
            of heralding checks to a state.

    """
    
    def __init__(self, circuit: Circuit, input_state: State, 
                 photon_counting: bool = True,
                 herald: FunctionType | None = None) -> None:
        
        # Assign parameters to attributes
        self.circuit = circuit
        self.input_state = input_state
        self.herald = herald
        self.photon_counting = photon_counting
        
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
    def herald(self) -> FunctionType:
        """A function to be used to post-selection of outputs."""
        return self.__herald
    
    @herald.setter
    def herald(self, value: FunctionType | None) -> None:
        if value is None:
            value = lambda s: True
        if type(value) != FunctionType:
            raise TypeError("Provided herald value should be a function.")
        self.__herald = value
        
    @property
    def photon_counting(self) -> bool:
        """Details whether photon number resolving detectors are in use."""
        return self.__photon_counting
        
    @photon_counting.setter
    def photon_counting(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Photon counting should be set to a boolean.")
        self.__photon_counting = bool(value)
        
    @property
    def probability_distribution(self):
        """
        The output probability distribution for the currently set configuration
        of the QuickSampler. This is re-calculated as the QuickSampler 
        parameters are changed.
        """
        if self._check_parameter_updates():
            # Check circuit and input modes match
            if self.circuit.n_modes != len(self.input_state):
                msg = "Mismatch in number of modes between input and circuit."
                raise ValueError(msg)
            # Check inputs are valid depending on the statistics and get type
            stats_type = self._check_stats_type()
            # For given input work out all possible outputs
            out_states = fock_basis(len(self.input_state), 
                                    self.input_state.num(), stats_type)
            if not self.photon_counting:
                out_states = [s for s in out_states if max(s) == 1]
            out_states = [s for s in out_states if self.herald(s)]
            if not out_states:
                msg = "Heralding function removed all possible outputs."
                raise ValueError(msg)
            # Find the probability distribution
            pdist = self._calculate_probabiltiies(out_states, stats_type)
            # Then assign calculated distribution to attribute
            self.__probability_distribution = pdist
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
    
    def sample_N(self, N: int, seed: int|None = None) -> list:
        """
        Function to sample from a circuit N times, this will return the 
        measured output count distribution.
        
        Args:
        
            N (int) : The number of samples to take from the circuit.
            
            seed (int|None, optional) : Option to provide a random seed to 
                reproducibly generate samples from the function. This is 
                optional and can remain as None if this is not required.
            
        Returns:
        
            list : A list containing the measured photon number counts for each
                of the modes.
        
        """
        # Create list to store output counts
        output_counts = [0]*self.circuit.n_modes
        pdist = self.probability_distribution
        vals = np.zeros(len(pdist), dtype=object)
        for i, k in enumerate(pdist.keys()):
            vals[i] = k
        # Generate N random samples and then process and count photons per mode
        np.random.seed(self._check_random_seed(seed))
        samples = np.random.choice(vals, p = list(pdist.values()), size = N)
        for state in samples:
            output_counts = [x + y for x, y in zip(output_counts, state)]
        return output_counts
    
    def sample_N_states(self, N: int, seed: int|None = None) -> SamplingResult:
        """
        Function to sample a state from the provided distribution many time, 
        if either heralding or post_select criteria are provided, the sampler 
        will only output states which meet these.
        
        Args:
        
            N (int) : The number of samples to take from the circuit.
            
            seed (int|None, optional) : Option to provide a random seed to 
                reproducibly generate samples from the function. This is 
                optional and can remain as None if this is not required.
        
        Returns:
        
            Result : A dictionary containing the different output states and 
                the number of counts for each one.
                    
        """
        pdist = self.probability_distribution
        vals = np.zeros(len(pdist), dtype=object)
        for i, k in enumerate(pdist.keys()):
            vals[i] = k
        # Generate N random samples and then process and count output states
        np.random.seed(self._check_random_seed(seed))
        samples = np.random.choice(vals, p = list(pdist.values()), size = N)
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
        if not hasattr(self, "_QuickSampler__calculation_values"):
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
        vals = [self.__circuit.U_full, self.input_state, self.herald, 
                self.photon_counting]
        return vals
    
    def _calculate_probabiltiies(self, outputs: list, stats_type: str) -> dict:
        """
        Calculate the probabilities of all of the provided outputs and returns
        this as a normalised distribution.
        """
        # Build circuit to avoid unnecessary recalculation of quantities
        built_circuit = self.circuit._build()
        # Add extra states on input for loss modes here when included
        in_state = self.input_state + State([0]*built_circuit.loss_modes)
        pdist = {}
        # Loop through possible outputs and calculate probability of each
        for ostate in outputs:
            p = Backend.calculate(built_circuit.U_full, in_state.s, 
                                    ostate + [0]*built_circuit.loss_modes, 
                                    stats_type)
            # Add output to distribution
            if abs(p)**2 > 0:
                pdist[State(ostate)] = abs(p)**2
        # Normalise probability distribution
        p_total = sum(pdist.values())
        for s, p in pdist.items():
            pdist[s] = p/p_total
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
    
    def _check_stats_type(self) -> str:
        """
        Returns the current stats type and also performs some checking that the
        currently set values are valid.
        """
        stats_type = get_statistic_type()
        if stats_type == "fermionic":
            self._fermionic_checks(self.input_state)
        return stats_type 
    
    def _fermionic_checks(self, in_state):
        """Perform additional checks when doing fermionic sampling."""
        if max(in_state) > 1:
            msg = """Max number of photons per mode must be 1 when using 
                        fermionic statistics."""
            raise ValueError(" ".join(msg.split()))
        
    def _check_random_seed(self, seed: Any) -> int | None:
        """Process a supplied random seed."""
        if not isinstance(seed, (int, type(None))):
            if int(seed) == seed:
                seed = int(seed)
            else:
                raise TypeError("Random seed must be an integer.")
        return seed
