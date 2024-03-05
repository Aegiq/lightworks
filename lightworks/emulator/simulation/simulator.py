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
from ..results import SimulationResult
from ...sdk import State, Circuit

import numpy as np


class Simulator:
    """
    Simulator class
    This class contains code required to simulate a circuit for a provided
    number of inputs and outputs. Currently the inputs/outputs should contain
    the same number of photons.
    
    Args:
    
        circuit : The circuit which is to be used for simulation.
        
    """
    def __init__ (self, circuit: Circuit) -> None:
        
        # Assign circuit to attribute
        self.circuit = circuit
        
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
    
    def simulate (self, inputs: list, 
                  outputs: list | None = None) -> SimulationResult:
        """
        Function to run a simulation for a number of inputs/outputs, if no 
        outputs are specified then all possible outputs for the photon number
        are calculated.
        
        Args:
        
            inputs (list) : A list of the input states to simulate. For 
                multiple inputs this should be a list of States.
            
            outputs (list | None, optional) : A list of the output states to 
                simulate, this can also be set to None to automatically find 
                all possible outputs.
            
        Returns:
        
            Result : A dictionary containing the calculated probability 
                amplitudes, where the first index of the array corresponds to 
                the input state, as well as the input and output state used to 
                create the array.
            
        """
        self.__stats_type = get_statistic_type()
        circuit = self.circuit._build()
        # Convert state to list of States if not provided for single state case
        if type(inputs) is State:
                inputs = [inputs]
        # Then process inputs list
        inputs = self._process_inputs(inputs)
        # And then either generate or process outputs
        inputs, outputs = self._process_outputs(inputs, outputs)
        # Add extra states for loss modes here when included
        if circuit.loss_modes > 0:
            inputs = [s + State([0]*circuit.loss_modes) for s in inputs]
            outputs = [s + State([0]*circuit.loss_modes) for s in outputs]
        # Calculate permanent for the given inputs and outputs and return 
        # values
        amplitudes = np.zeros((len(inputs), len(outputs)), dtype = complex)
        for i, ins in enumerate(inputs):
            for j, outs in enumerate(outputs):
                amplitudes[i, j] = Backend.calculate(circuit.U_full, 
                                                     ins.s, outs.s, 
                                                     self.__stats_type)
        if circuit.loss_modes > 0:
            inputs = [s[:circuit.n_modes] for s in inputs]
            outputs = [s[:circuit.n_modes] for s in outputs]
        # Return results and corresponding states as dictionary
        results = {"amplitudes" : amplitudes, "inputs" : inputs, 
                   "outputs" : outputs}
        results = SimulationResult(amplitudes, "probability_amplitude", 
                                   inputs = inputs, outputs = outputs)
        return results
    
    def _process_inputs(self, inputs: list) -> list:
        """Performs all required processing/checking on the input states."""
        # Check each input
        for state in inputs:
            # Ensure correct type
            if not isinstance(state, State):
                msg = "inputs should be a State or list of State objects."
                raise TypeError(msg)
            # Fermionic checks
            if self.__stats_type == "fermionic":
                if max(state) > 1:
                    msg = """Max number of photons per mode must be 1 when
                             using fermionic statistics."""
                    raise ValueError(" ".join(msg.split()))   
            # Dimension check
            if len(state) != self.circuit.n_modes:
                msg = f"""One or more input states have an incorrect number of 
                          modes, correct number of modes is 
                          {self.circuit.n_modes}."""
                raise StateError(" ".join(msg.split())) 
        return inputs
    
    def _process_outputs(self, inputs: list, 
                         outputs: list | None) -> tuple[list, list]:
        """
        Processes the provided outputs or generates them if no inputs were 
        provided. Returns both the inputs and outputs.
        """
        # If outputs not specified then determine all combinations
        if outputs is None:
            ns = [s.num() for s in inputs]
            if min(ns) != max(ns):
                msg = """Mismatch in total photon number between inputs, this 
                         is not currently supported by the Simulator."""
                raise StateError(" ".join(msg.split()))
            outputs = fock_basis(self.circuit.n_modes, max(ns), 
                                 self.__stats_type)
            outputs = [State(s) for s in outputs]
        # Otherwise check provided outputs
        else:
            if type(outputs) is State:
                outputs = [outputs]
            # Check type and for fermionic statistics
            for state in outputs:
                # Ensure correct type
                if not isinstance(state, State):
                    msg = "outputs should be a State or list of State objects."
                    raise TypeError(msg)  
                # Fermionic checks
                if self.__stats_type == "fermionic":
                    if max(state) > 1:
                        msg = """Max number of photons per mode must be 1 when
                                using fermionic statistics."""
                        raise ValueError(" ".join(msg.split()))
                # Dimension check
                if len(state) != self.circuit.n_modes:
                    msg = f"""One of more input states have an incorrect number
                              of modes, correct number of modes is 
                              {self.circuit.n_modes}."""
                    raise StateError(" ".join(msg.split())) 
            # Ensure photon numbers are the same in all states - variation not 
            # currently supported
            ns = [s.num() for s in inputs + outputs]
            if min(ns) != max(ns):
                msg = """Mismatch in photon numbers between some 
                         inputs/outputs, this is not currently supported in the 
                         simulator."""
                raise StateError(" ".join(msg.split()))
        return inputs, outputs