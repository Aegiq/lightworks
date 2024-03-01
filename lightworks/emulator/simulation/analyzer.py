from .backend import Backend
from ..utils import fock_basis, get_statistic_type
from ..utils import StateError
from ..results import SimulationResult
from ...sdk import State, Circuit

import numpy as np
from types import FunctionType


class Analyzer:
    """
    Analyzer class
    
    The analyzer class is built as an alternative to simulation, intended for
    cases where we want to look at the transformations between a specific 
    subset of states. It is useful for the simulation of probabilities in 
    cases where loss and circuit errors are likely to be a factor. As part of
    the process a performance and error rate metric are calculated.  
    
    Args:
    
        circuit : The circuit to simulate.
        
    Attribute:
    
        performance : The total probabilities of mapping between the states 
            provided compared with all possible states.
                      
        error_rate : Given an expected mapping, the analyzer will determine the
            extent to which this is achieved.
        
    """
    def __init__ (self, circuit: Circuit) -> None:
        
        # Assign key parameters to attributes
        self.circuit = circuit
        # Create empty list/dict to store other quantities
        self.in_heralds = {}
        self.out_heralds = {}
        self.post_selects = []
        
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
    
    def set_herald(self, in_mode: int, n_photons: int = 0, 
                   out_mode: int | None = None) -> None:
        """
        Set heralded modes for the circuit
        """
        # Check n_photons <= 1 for fermionic case
        if get_statistic_type() == "fermionic":
            if n_photons > 1:
                msg = """Heralding photons should be one or less when using 
                         fermionic statistics."""
                raise ValueError(" ".join(msg.split()))
        if out_mode is None:
            out_mode = in_mode
        if type(in_mode) != int or type(out_mode) != int:
            raise TypeError("Mode numbers should be integers.")
        if in_mode >= self.circuit.n_modes or out_mode >= self.circuit.n_modes:
            raise ValueError("Mode outside of circuit range.")
        self.in_heralds[in_mode] = n_photons
        self.out_heralds[out_mode] = n_photons
        
        return
    
    def set_post_selection(self, function: FunctionType) -> None:
        """
        Add post selection functions, these should apply to non-heralded modes.
        """
        # NOTE: If multiple lambda functions are created and passed to this 
        # using a loop this may create issues related to how lambda functions
        # use out of scope variables. See the following for more info:
        # https://docs.python.org/3/faq/programming.html#why-do-lambdas-
        # defined-in-a-loop-with-different-values-all-return-the-same-result
        if not isinstance(function, FunctionType):
            raise TypeError("Post-selection rule should be a function.")
        self.post_selects += [function]
        
        return
    
    def analyze (self, inputs: State | list, 
                 expected: dict | None = None) -> SimulationResult:
        """
        Function to perform analysis of probabilities between 
        different inputs/outputs
        
        Args:
        
            inputs (list) : A list of the input states to simulate. For 
                multiple inputs this should be a list of States.
                            
            expected (dict) : A dictionary containing a mapping between the
                input state and expected output state(s). If there is multiple
                possible outputs, this can be specified as a list.
                        
        Returns:
        
            dict : A dictionary containing an array of probability values 
                between the provided inputs/outputs. 
            
        """
        self.__stats_type = get_statistic_type()
        self.__circuit_built = self.circuit._build()
        n_modes = self.__circuit_built.n_modes - len(self.in_heralds)
        if self.in_heralds != self.out_heralds:
            msg = """Mismatch in number of heralds on the input/output modes,
                     it is likely this results from a herald being added twice
                     or modified."""
            raise StateError(" ".join(msg.split()))
        # Convert state to list of States if not provided for single state case
        if type(inputs) is State:
                inputs = [inputs] 
        # Process inputs using dedicated function
        full_inputs = self._process_inputs(inputs)
        n_photons = sum(full_inputs[0])
        # Generate lists of possible outputs with and without heralded modes
        full_outputs, filtered_outputs = self._generate_outputs(n_modes, 
                                                                n_photons)
        # Calculate permanent for the given inputs and outputs and return 
        # values
        probs = self._get_probs(full_inputs, full_outputs)
        # Calculate performance by finding sum of valid transformations
        self.performance = probs.sum()/len(full_inputs)
        # Analyse error rate from expected results if specified
        if expected is not None:
            self.error_rate = self._calculate_error_rate(probs, inputs, 
                                                         filtered_outputs,
                                                         expected)
        # Compile results into results object
        results = SimulationResult(probs, "probability", inputs = inputs,
                                   outputs = filtered_outputs, 
                                   performance = self.performance)
        if hasattr(self, "error_rate"):
            results.error_rate = self.error_rate
        self.results = results
        # Return dict
        return results
    
    def _get_probs(self, full_inputs: list, full_outputs: list) -> np.ndarray:
        """
        Create an array of output probabilities for a given set of inputs and 
        outputs.
        """
        probs = np.zeros((len(full_inputs), len(full_outputs)))
        for i, ins in enumerate(full_inputs):
            for j, outs in enumerate(full_outputs):
                # No loss case
                if not self.__circuit_built.loss_modes:
                    p = Backend.calculate(self.__circuit_built.U_full, ins.s, 
                                          outs.s, self.__stats_type)
                    probs[i, j] += abs(p)**2
                # Lossy case
                else:
                    # Photon number preserved
                    if ins.num() == outs.num():
                        outs = (outs + 
                                State([0]*self.__circuit_built.loss_modes))
                        p = Backend.calculate(self.__circuit_built.U_full, 
                                              ins.s, outs.s, self.__stats_type)
                        probs[i, j] += abs(p)**2
                    # Otherwise
                    else:
                        # If n_out < n_in work out all loss mode combinations
                        # and find probability of each
                        n_loss = ins.num() - outs.num()
                        if n_loss < 0:
                            msg = """Output photon number larger than input 
                                     number."""
                            raise StateError(" ".join(msg.split()))
                        # Find loss states and find probability of each
                        loss_states = fock_basis(
                            self.__circuit_built.loss_modes, n_loss, 
                            self.__stats_type
                            )
                        for ls in loss_states:
                            fs = outs + State(ls)
                            p = Backend.calculate(self.__circuit_built.U_full, 
                                                  ins.s, fs.s, 
                                                  self.__stats_type)
                            probs[i, j] += abs(p)**2       
        return probs
    
    def _build_state(self, state: State, herald: FunctionType) -> State:
        """
        Add heralded modes to a provided state
        """
        count = 0
        new_state = [0]*self.__circuit_built.n_modes
        for i in range(self.__circuit_built.n_modes):
            if i in herald:
                new_state[i] = herald[i]
            else:
                new_state[i] = state[count]
                count += 1
        
        return State(new_state)
    
    def _calculate_error_rate(self, probabilities: np.ndarray, inputs: list, 
                              outputs: list, expected: dict) -> float:
        """
        Calculate the error rate for a set of expected mappings between inputs 
        and outputs, given the results calculated by the analyzer.
        """
        # Check all inputs in expectation mapping
        for s in inputs:
            if s not in expected:
                msg = f"Input state {s} not in provided expectation dict."
                raise StateError(msg)
        # For each input check error rate
        errors = []
        for i, s in enumerate(inputs):
            out = expected[s]
            # Convert expected output to list if only one value provided
            if type(out) == State:
                out = [out]
            iprobs = probabilities[i,:]
            error = 1
            # Loop over expected outputs and subtract from error value
            for o in out:
                if o in outputs:
                    loc = outputs.index(o)
                    error -= (iprobs[loc]/sum(iprobs))
            errors += [error]
        # Then take average and return
        return np.mean(errors)
    
    def _process_inputs(self, inputs: list) -> list:
        """
        Takes the provided inputs, perform error checking on them and adds any 
        heralded photons that are required, returning full states..
        """
        n_modes = self.__circuit_built.n_modes - len(self.in_heralds)
        # Ensure all photon numbers are the same   
        ns = [s.num() for s in inputs]
        if min(ns) != max(ns):
            raise StateError("Mismatch in photon numbers between inputs")
        full_inputs = []
        # Check dimensions of input and add heralded photons
        for state in inputs:
            # Also check input state is valid in fermionic case
            if self.__stats_type == "fermionic":
                if max(state) > 1:
                    msg = """Max number of photons per mode must be 1 when 
                             using fermionic statistics."""
                    raise ValueError(" ".join(msg.split()))
            if len(state) != n_modes:
                msg = ("Input states are of the wrong dimension. " + 
                       "Remember to subtract Heralded modes.")
                raise StateError(msg)
            full_inputs += [self._build_state(state, self.in_heralds)]
        # Add extra states for loss modes here when included
        if self.__circuit_built.loss_modes > 0:
            full_inputs = [s + State([0]*self.__circuit_built.loss_modes) 
                           for s in full_inputs]
        return full_inputs
    
    def _generate_outputs(self, n_modes: int, 
                          n_photons: int) -> tuple[list, list]:
        """
        Generates all possible outputs for a given number of modes, photons and
        heralding + post-selection conditions. It returns two list, one with 
        the heralded modes included and one without.
        """
        # Get all possible outputs for the non-herald modes
        if not self.__circuit_built.loss_modes:            
            outputs = fock_basis(n_modes, n_photons, self.__stats_type)
        # Combine all n < n_in for lossy case
        else:
            outputs = []
            for n in range(0, n_photons+1):
                outputs += fock_basis(n_modes, n, self.__stats_type)
            # Remove invalid fermionic outputs in case where this is not enough
            # loss modes to support a potential lossy output
            if self.__stats_type == "fermionic":
                outputs = [o for o in outputs if 
                           sum(o)+self.__circuit_built.loss_modes >= n_photons]
        # Filter outputs according to post selection and add heralded photons
        filtered_outputs = []
        full_outputs = []
        for state in outputs:
            fo  = self._build_state(state, self.out_heralds)
            # Check output meets all post selection rules
            for funcs in self.post_selects:
                if not funcs(fo):
                    break
            else:
                filtered_outputs += [State(state)]
                full_outputs += [fo]
        # Check some valid outputs found
        if not full_outputs:
            msg = "No valid outputs found, consider relaxing post-selection."
            raise ValueError(msg)
        
        return (full_outputs, filtered_outputs)