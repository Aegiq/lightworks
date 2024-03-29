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
from ..utils import fock_basis
from ..utils import ModeMismatchError, PhotonNumberError
from ..results import SimulationResult
from ...sdk import State, Circuit

import numpy as np


class Simulator:
    """
    Simulates a circuit for a provided number of inputs and outputs, returning
    the probability amplitude between them.
    
    Args:
    
        circuit : The circuit which is to be used for simulation.
        
    """
    def __init__ (self, circuit: Circuit) -> None:
        
        # Assign circuit to attribute
        self.circuit = circuit
        self.__backend = Backend("permanent")
        
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
            raise TypeError(
                "Provided circuit should be a Circuit or Unitary object.")
        self.__circuit = value
    
    def simulate (self, inputs: list, 
                  outputs: list | None = None) -> SimulationResult:
        """
        Function to run a simulation for a number of inputs/outputs, if no 
        outputs are specified then all possible outputs for the photon number
        are calculated. All inputs and outputs should have the same photon 
        number.
        
        Args:
        
            inputs (list) : A list of the input states to simulate. For 
                multiple inputs this should be a list of States.
            
            outputs (list | None, optional) : A list of the output states to 
                simulate, this can also be set to None to automatically find 
                all possible outputs.
            
        Returns:
        
            SimulationResult : A dictionary containing the calculated 
                probability amplitudes, where the first index of the array 
                corresponds to the input state, as well as the input and output
                state used to create the array.
            
        """
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
                amplitudes[i, j] = self.__backend.probability_amplitude(
                    circuit.U_full, ins.s, outs.s)
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
                raise TypeError(
                    "inputs should be a State or list of State objects.")  
            # Dimension check
            if len(state) != self.circuit.n_modes:
                raise ModeMismatchError(
                    "One or more input states have an incorrect number of "
                    "modes, correct number of modes is "
                    f"{self.circuit.n_modes}.") 
        return inputs
    
    def _process_outputs(self, inputs: list, 
                         outputs: list | None) -> tuple[list, list]:
        """
        Processes the provided outputs or generates them if no inputs were 
        provided. Returns both the inputs and outputs.
        """
        # If outputs not specified then determine all combinations
        if outputs is None:
            ns = [s.n_photons for s in inputs]
            if min(ns) != max(ns):
                raise PhotonNumberError(
                    "Mismatch in total photon number between inputs, this is "
                    "not currently supported by the Simulator.")
            outputs = fock_basis(self.circuit.n_modes, max(ns))
            outputs = [State(s) for s in outputs]
        # Otherwise check provided outputs
        else:
            if type(outputs) is State:
                outputs = [outputs]
            # Check type and for fermionic statistics
            for state in outputs:
                # Ensure correct type
                if not isinstance(state, State):
                    raise TypeError(
                        "outputs should be a State or list of State objects.")  
                # Dimension check
                if len(state) != self.circuit.n_modes:
                    raise ModeMismatchError(
                        "One or more input states have an incorrect number of "
                        "modes, correct number of modes is "
                        f"{self.circuit.n_modes}.")
            # Ensure photon numbers are the same in all states - variation not 
            # currently supported
            ns = [s.n_photons for s in inputs + outputs]
            if min(ns) != max(ns):
                raise PhotonNumberError(
                    "Mismatch in photon numbers between some inputs/outputs, "
                    "this is not currently supported in the Simulator.")
        return inputs, outputs