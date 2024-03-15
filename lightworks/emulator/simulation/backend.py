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

from .permanent import Permanent
from .slos import SLOS
from ..utils import fock_basis
from ...sdk import Circuit, State

from numpy import ndarray

class Backend:
    """
    Provide central location for selecting and interacting with different 
    simulation backends.
    """
    
    def __init__(self, backend: str) -> None:
        
        self.backend = backend
        
        return
    
    @property
    def backend(self) -> str:
        """Stores data on the selected backend."""
        return self.__backend
    
    @backend.setter
    def backend(self, value: str) -> None:
        if value not in ["permanent", "slos", "clifford"]:
            raise ValueError("Invalid backend provided.")
        # Temporary extra checking
        if value in ["clifford"]:
            raise NotImplementedError(
                "Support for this backend has not yet been included.")
        self.__backend = value
        
    def full_probability_distribution(self, circuit: Circuit, 
                                      input_state: State) -> dict:
        """Finds the probability distribution for the provided input state."""
        
        if self.backend == "permanent":    
            # Add extra states for loss modes here when included
            if circuit.loss_modes > 0:
                input_state = input_state + State([0]*circuit.loss_modes) 
            pdist = {}
            # For a given input work out all possible outputs
            out_states = fock_basis(len(input_state), input_state.n_photons)
            for ostate in out_states:
                # Skip any zero photon states
                if sum(ostate[:circuit.n_modes]) == 0:
                    continue
                p = Permanent.calculate(circuit.U_full, input_state.s, ostate)
                if abs(p)**2 > 0:
                    # Only care about non-loss modes
                    ostate = State(ostate[:circuit.n_modes])
                    if ostate in pdist:
                        pdist[ostate] += abs(p)**2
                    else:
                        pdist[ostate] = abs(p)**2
            # Work out zero photon component before saving to unique results
            total_prob = sum(pdist.values())
            if total_prob < 1 and circuit.loss_modes > 0:
                pdist[State([0]*circuit.n_modes)] = 1 - total_prob
        elif self.backend == "slos":
            # Add extra states for loss modes here when included
            if circuit.loss_modes > 0:
                input_state = input_state + State([0]*circuit.loss_modes) 
            full_dist = SLOS.calculate(circuit.U_full, input_state)
            # Combine results to remote lossy modes
            pdist = {}
            for s, p in full_dist.items():
                new_s = State(s[:circuit.n_modes])
                if new_s in pdist:
                    pdist[new_s] += p
                else:
                    pdist[new_s] = p
        elif self.backend == "clifford": 
            raise RuntimeError(
                "Probability distribution calculation not supported for "
                "clifford backend.")
        
        return pdist