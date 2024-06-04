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

from ..backend import Backend
from ...sdk.state import State
from ...sdk.circuit.circuit_compiler import CompiledCircuit

class ProbabilityDistributionCalc:
    
    @staticmethod
    def state_prob_calc(circuit: CompiledCircuit, inputs: dict,
                        backend: Backend) -> dict:
        """
        Calculate the output state probability distribution for cases where 
        inputs are state objects. This is the case when the source is perfect
        or only an imperfect brightness is used.
        
        Args:
        
            circuit (CompiledCircuit) : The compiled circuit that is to be 
                sampled from.
                                        
            inputs (dict) : The inputs to the system and their associated 
                probabilities. 
                            
        Returns:
        
            dict : The calculated output probability distribution.
            
        """
        pdist = {}
        # Loop over each possible input
        for istate, prob in inputs.items():
            # Calculate sub distribution
            sub_dist = backend.full_probability_distribution(circuit, istate)
            if not pdist:
                if prob == 1:
                    pdist = sub_dist
                else:
                    pdist = {s : p*prob for s, p in sub_dist.items()}
            else:
                for s, p in sub_dist.items():
                    if s in pdist:
                        pdist[s] += p*prob
                    else:
                        pdist[s] = p*prob
        # Calculate zero photon state probability afterwards
        total_prob = sum(pdist.values())
        if total_prob < 1 and circuit.loss_modes > 0:
            pdist[State([0]*circuit.n_modes)] = 1 - total_prob
        
        return pdist
    
    @staticmethod
    def annotated_state_prob_calc(circuit: CompiledCircuit, inputs: dict, 
                                  backend: Backend) -> dict:
        """
        Perform output state probability distribution calculation using complex
        annotated states, with imperfect purity and/or indistinguishability.
        
        Args:
        
            circuit (CompiledCircuit) : The compiled circuit that is to be 
                                        sampled from.
                                        
            inputs (dict) : The inputs to the system and their associated 
                            probabilities. 
                            
        Returns:
        
            dict : The calculated output probability distribution.
            
        """
        # Determine the input state combinations given the labels
        unique_inputs = set()
        input_combinations = {}
        for state, p in inputs.items():
            # Find all labels in a given state
            all_labels = []
            for mode in state: 
                all_labels += mode
            # For all labels break them down into the corresponding states
            if all_labels:
                results = {l:[0]*circuit.n_modes for l in all_labels}
                for i, mode in enumerate(state):
                    for m in mode:
                        results[m][i] += 1
                states = [State(s) for s in results.values()]
                unique_inputs = unique_inputs | set(states)
            else: # Special case for empty annotated state
                states = [State([0]*circuit.n_modes)]
                unique_inputs.add(State([0]*circuit.n_modes))
            input_combinations[state] = states            
        # For each of the unique inputs then need to work out the probability 
        # distribution
        unique_results = {}
        for istate in unique_inputs:
            # Calculate sub distribution and store
            unique_results[istate[:circuit.n_modes]] = (
                backend.full_probability_distribution(circuit, istate))
        
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
                                new_pdist[new_state] = p1*p2
                            else:
                                new_pdist[new_state] += p1*p2
                    pdist = new_pdist
            # Then combine outputs and weight by input probability        
            ip = inputs[istate]
            for ostate, op in pdist.items():
                if ostate not in stats_dict:
                    stats_dict[ostate] = ip*op
                else:
                    stats_dict[ostate] += ip*op

        return stats_dict