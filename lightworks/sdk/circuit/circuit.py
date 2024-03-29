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

"""
Circuit class for creating circuits with Parameters object that can be modified
after creation.
"""

from .circuit_compiler import CompiledCircuit, CompiledUnitary
from .parameters import Parameter
from ..utils import ModeRangeError, DisplayError
from ..utils import CircuitCompilationError
from ..utils import unpack_circuit_spec, compress_mode_swaps
from ..utils import convert_non_adj_beamsplitters
from ..visualisation import Display

from typing import Any, Union
from numbers import Number
import matplotlib.pyplot as plt
from IPython import display
from copy import copy, deepcopy
import numpy as np


class Circuit:
    """
    Provides support for building circuits from a set of linear optic 
    components, with the ability to assign certain quantities of components to 
    Parameter objects whose values can be adjusted after creation. 
    
    Args:
    
        n_modes (int) : The number of modes used in the circuit.
                         
    """
    
    def __init__(self, n_modes: int) -> None:
        
        if not isinstance(n_modes, int):
            if int(n_modes) == n_modes:
                n_modes = int(n_modes)
            else:
                raise TypeError("Number of modes should be an integer.")
        self.__n_modes = n_modes
        self.__circuit_spec = []
        self.__show_parameter_values = False
        
        return
    
    def __add__(self, value: "Circuit") -> "Circuit":
        """Defines custom addition behaviour for two circuits."""
        if isinstance(value, Circuit):
            if self.__n_modes != value.__n_modes:
                raise ModeRangeError(
                    "Two circuits to add must have the same number of modes.")
            # Create new circuits and combine circuits specs to create new one
            new_circ = Circuit(self.__n_modes)
            new_circ.__circuit_spec = (self.__circuit_spec + 
                                       value.__circuit_spec)
            return new_circ
        else:
            raise TypeError(
                "Addition only supported between two Circuit objects.")
        
    @property
    def U(self) -> np.ndarray:
        """
        The effective unitary that the circuit implements across modes. This 
        will include the effect of any loss within a circuit. It is calculated 
        using the parameter values at the time that the attribute is called.
        """
        return self._build().U_full[:self.__n_modes, :self.__n_modes]
    
    @property
    def U_full(self) -> np.ndarray:
        """
        The full unitary for the created circuit, this will include the 
        additional modes used for the simulation of loss, if this has been 
        included in the circuit.
        """
        return self._build().U_full
    
    @property
    def n_modes(self) -> int:
        """Returns number of modes in the circuit."""
        return self.__n_modes
    
    @n_modes.setter
    def n_modes(self, value: Any) -> None:
        """
        Prevents modification of n_modes attribute after circuit creation.
        """
        raise AttributeError("Number of circuit modes cannot be modified.")
    
    @property
    def _display_spec(self) -> list:
        """Returns display spec for the circuit."""
        return self._get_display_spec()
        
    def add(self, circuit: Union["Circuit", "Unitary"],                        # type: ignore - ignores Pylance warning raised by undefined unitary component
            mode: int = 0, group: bool = False, name: str = "Circuit") -> None:
        """
        Can be used to add either another Circuit or a Unitary component to the
        existing circuit. This can either have the same size or be smaller than
        the circuit which is being added to.
        
        Args:
        
            circuit (Circuit | Unitary) : The circuit/component that is to be 
                added.
                                                       
            mode (int, optional) : The first mode on which the circuit should 
                be placed. If not specified it defaults to zero.
        """
        if isinstance(circuit, Circuit):
            self._mode_in_range(mode)
            if mode + circuit.__n_modes > self.__n_modes:
                raise ModeRangeError("Circuit to add is outside of mode range")
            if group:
                spec = unpack_circuit_spec(circuit.__circuit_spec)
            else:
                spec = circuit.__circuit_spec
            add_cs = self._add_mode_to_circuit_spec(spec, mode)
            if not group:
                self.__circuit_spec = self.__circuit_spec + add_cs
            else:
                self.__circuit_spec.append(["group", (add_cs, name, mode, 
                                                      mode + 
                                                      circuit.__n_modes - 1)])
        else:
            raise TypeError(
                "Add method only supported for Circuit or Unitary objects.")
        
        return
    
    def add_bs(self, mode_1: int, mode_2: int = None, 
               reflectivity: float = 0.5, loss: float = 0,
               convention: str = "Rx") -> None:
        """
        Add a beam splitter of specified reflectivity between two modes to the 
        circuit.
        
        Args:
        
            mode_1 (int) : The first mode which the beam splitter acts on.
            
            mode_2 (int | None, optional) : The second mode that the beam 
                splitter acts on. This can also be left as the default value of
                None to automatically use mode_2 as mode_1 + 1.
            
            reflectivity (float, optional) : The reflectivity value of the
                beam splitter. Defaults to 0.5.
            
            loss (float, optional) : The loss of the beam splitter, this should
                be positive and given in dB.
                                     
            convention (str, optional) : The convention to use for the beam 
                splitter, should be either "Rx" (the default value) or "H".
                
        """
        self._mode_in_range(mode_1)
        if mode_2 is None: mode_2 = mode_1 + 1
        if mode_1 == mode_2:
            raise ModeRangeError(
                "Beam splitter must act across two different modes.")
        self._mode_in_range(mode_2)
        self._check_loss(loss)
        # Check correct convention is given
        all_convs = ["Rx", "H"]
        if convention not in all_convs:
            msg = "Provided beam splitter convention should in ['"
            for conv in all_convs:
                msg += conv + ", "
            msg = msg[:-2] + "']"
            raise ValueError(msg)
        self.__circuit_spec.append(["bs", (mode_1, mode_2, reflectivity, 
                                           convention)])
        if isinstance(loss, Parameter) or loss > 0:
            self.add_loss(mode_1, loss)
            self.add_loss(mode_2, loss)
        
    def add_ps(self, mode: int, phi: float, loss: float = 0) -> None:
        """
        Applies a phase shift to a given mode in the circuit.
        
        Args:
        
            mode (int) : The mode on which the phase shift acts.
            
            phi (float) : The angular phase shift to apply.
            
            loss (float, optional) : The loss of the phase shifter, this should
                be positive and given in dB.
                                                   
        """
        self._mode_in_range(mode)
        self._check_loss(loss)
        self.__circuit_spec.append(["ps", (mode, phi)])
        if isinstance(loss, Parameter) or loss > 0:
            self.add_loss(mode, loss)
        
    def add_loss(self, mode: int, loss: float = 0) -> None:
        """
        Adds a loss channel to the specified mode of the circuit.
        
        Args:
        
            mode (int) : The mode on which the loss channel acts.
                        
            loss (float, optional) : The loss applied to the selected mode, 
                this should be positive and given in dB.
                                               
        """
        self._mode_in_range(mode)
        self._check_loss(loss)
        self.__circuit_spec.append(["loss", (mode, loss)])
        
    def add_barrier(self, modes: list | None = None) -> None:
        """
        Adds a barrier to separate different parts of a circuit. This is 
        applied to the specified modes.
        
        Args:
        
            modes (list | None) : The modes over which the barrier should be
                applied to.
                
        """
        if modes == None:
            modes = [i for i in range(self.__n_modes)]
        for m in modes:
            self._mode_in_range(m)
        self.__circuit_spec.append(["barrier", tuple([modes])])
        
    def add_mode_swaps(self, swaps: dict) -> None:
        """
        Performs swaps between a given set of modes. The keys of the dictionary
        should correspond to the initial modes and the values to the modes they
        are swapped to. It is also required that all mode swaps are complete,
        i.e. any modes which are swapped to must also be sent to a target 
        destination.
        
        Args:
        
            swaps (dict) : A dictionary detailing the original modes and the 
                locations that they are to be swapped to. 
                           
        """
        # Perform some error checking steps
        in_modes = sorted(list(swaps.keys()))
        out_modes = sorted(list(swaps.values()))
        if in_modes != out_modes:
            raise ValueError(
                "Mode swaps not complete, dictionary keys and values should "
                "contain the same modes.")
        for m in in_modes:
            self._mode_in_range(m)
        self.__circuit_spec.append(["mode_swaps", (swaps, None)])
        
    def display(self, show_parameter_values: bool = False, 
                display_loss: bool = False, 
                mode_labels: list | None = None,
                display_type: str = "svg") -> None:
        """
        Displays the current circuit with parameters set using either their 
        current values or labels.
        
        Args:
        
            show_parameter_values (bool, optional) : Shows the values of 
                parameters instead of the associated labels if specified.
        
            display_loss (bool, optional) : Choose whether to display loss
                components in the figure, defaults to False.
                                            
            mode_labels (list|None, optional) : Optionally provided a list of 
                mode labels which will be used to name the mode something other
                than numerical values. Can be set to None to use default 
                values.
                                                
            display_type (str, optional) : Selects whether the drawsvg or 
                matplotlib module should be used for displaying the circuit. 
                Should either be 'svg' or 'mpl', defaults to 'svg'.
                
        """
        
        self.__show_parameter_values = show_parameter_values
        return_ = Display(self, display_loss = display_loss, 
                          mode_labels = mode_labels,
                          display_type = display_type)
        if display_type == "mpl":
            plt.show()
        elif display_type == "svg":
            display.display(return_)
        # Return show parameter value attribute to default value after display
        self.__show_parameter_values = False
        return
    
    def get_all_params(self) -> list:
        """
        Returns all the Parameter objects used as part of creating the circuit.
        """
        all_params = []
        for _, params in unpack_circuit_spec(deepcopy(self.__circuit_spec)):
            for p in params:
                if isinstance(p, Parameter) and p not in all_params:
                    all_params.append(p)
        return all_params    
    
    def copy(self, freeze_parameters: bool =  False) -> "Circuit":
        """
        Creates and returns an identical copy of the circuit, optionally 
        freezing parameter values.
        
        Args:
        
            freeze_parameters (bool, optional) : Used to control where any 
                existing parameter objects are carried over to the newly 
                created circuit, or if the current parameter values should be 
                used. Defaults to False.
            
        Returns:
        
            Circuit : A new Circuit object with the same circuit configuration 
                as the original object.
                                
        """
        new_circ = Circuit(self.n_modes)
        if not freeze_parameters:
            new_circ.__circuit_spec = copy(self.__circuit_spec)
        else:
            copied_spec = deepcopy(self.__circuit_spec)
            new_circ.__circuit_spec = self._freeze_params(copied_spec)
        return new_circ
    
    def unpack_groups(self) -> None:
        """
        Unpacks any component groups which may have been added to the circuit.
        """
        self.__circuit_spec = unpack_circuit_spec(self.__circuit_spec)
        return
        
    def compress_mode_swaps(self) -> None:
        """
        Takes a provided circuit spec and will try to compress any more swaps
        such that the circuit length is reduced. Note that any components in a 
        group will be ignored.    
        """
        # Convert circuit spec and then assign to attribute
        new_spec = compress_mode_swaps(deepcopy(self.__circuit_spec))
        self.__circuit_spec = new_spec
        
    def remove_non_adjacent_bs(self) -> None:
        """
        Removes any beam splitters acting on non-adjacent modes by replacing
        with a mode swap and adjacent beam splitters. 
        """
        # Convert circuit spec and then assign to attribute
        spec = deepcopy(self.__circuit_spec)
        new_spec = convert_non_adj_beamsplitters(spec)
        self.__circuit_spec = new_spec
    
    def _build(self) -> CompiledCircuit:
        """
        Converts the ParameterizedCircuit into a circuit object using the 
        components added and current values of the parameters.
        """
        
        try:
            circuit = self._build_process()
        except Exception as e:
            msg = "An error occurred during the circuit compilation process"
            raise CircuitCompilationError(msg) from e
        
        return circuit
    
    def _build_process(self) -> CompiledCircuit:
        """
        Contains full process for convert a circuit into a compiled one.
        """
        circuit = CompiledCircuit(self.__n_modes)
        
        for cs, params in unpack_circuit_spec(self.__circuit_spec):
            got_params = [p.get() if isinstance(p, Parameter) else p 
                          for p in params]
            if cs == "bs":
                circuit.add_bs(*got_params)
            elif cs == "ps":
                circuit.add_ps(*got_params)
            elif cs == "loss":
                circuit.add_loss(*got_params)
            elif cs == "barrier":
                circuit.add_barrier(got_params[0])
            elif cs == "mode_swaps":
                circuit.add_mode_swaps(got_params[0])
            elif cs == "unitary":
                unitary = CompiledUnitary(params[1])
                circuit.add(unitary, params[0])
            else:
                raise CircuitCompilationError(
                    "Component in circuit spec not recognised.")
        
        return circuit
    
    def _mode_in_range(self, mode: int) -> bool:
        """
        Check a mode exists within the created circuit and also confirm it is 
        an integer.
        """
        if isinstance(mode, Parameter):
            raise TypeError("Mode values cannot be parameters.")
        # Catch this separately as bool is subclass of int
        elif isinstance(mode, bool): 
            raise TypeError("Mode number should be an integer.")
        elif not isinstance(mode, int):
            if int(mode) == mode:
                mode = int(mode)
            else:
                raise TypeError("Mode number should be an integer.")
        if 0 <= mode < self.__n_modes:
            return True
        else:
            raise ModeRangeError(
                "Selected mode(s) is not within the range of the created "
                "circuit.")
        
    def _check_loss(self, loss: float) -> bool:
        """
        Check that loss is positive and toggle _loss_included if not already 
        done.
        """
        if isinstance(loss, Parameter):
            loss = loss.get()
        if not isinstance(loss, Number) or isinstance(loss, bool):
            raise TypeError("Loss value should be numerical or a Parameter.")
        if loss < 0:
            raise ValueError("Provided loss values should be greater than 0.")
        else:
            return True
        
    def _add_mode_to_circuit_spec(self, circuit_spec: list, mode: int) -> list:
        """
        Takes an existing circuit spec and adds a given number of modes to each
        of the elements.
        """
        new_circuit_spec = []
        for c, params in circuit_spec:
            params = list(params)
            if c in ["bs"]:
                params[0] = params[0] + mode
                params[1] = params[1] + mode
            elif c == "barrier":
                params = [p+mode for p in params[0]]
                params = tuple([params])
            elif c == "mode_swaps":
                params[0] = {k+mode:v+mode for k,v in params[0].items()}
            elif c == "group":
                params[0] = self._add_mode_to_circuit_spec(params[0], mode)
                params[2] += mode
                params[3] += mode
            else:
                params[0] = params[0] + mode
            new_circuit_spec.append([c, tuple(params)])
        return new_circuit_spec
        
    def _get_display_spec(self) -> list:
        """
        Creates a parameterized version of the circuit build spec using the 
        included components.
        """
        
        display_spec = []
        for cs, params in self.__circuit_spec:
            # Convert parameters into labels if specified or values if not 
            if not self.__show_parameter_values:
                cparams = [p.label if (isinstance(p, Parameter) 
                           and p.label is not None) else p.get() 
                           if isinstance(p, Parameter) else p for p in params]
            else:
                cparams = [p.get() if isinstance(p, Parameter) else p 
                           for p in params]
            # Adjust build spec for each component
            if cs == "bs":
                display_spec.append(("BS", [cparams[0], cparams[1]], 
                                   (cparams[2])))
            elif cs == "ps":
                display_spec.append(("PS", cparams[0], (cparams[1])))
            elif cs == "loss":
                if self._display_loss_check(cparams[1]):
                    display_spec.append(("LC", cparams[0], (cparams[1])))
            elif cs == "barrier":
                display_spec.append(("barrier", cparams[0], None))
            elif cs == "mode_swaps":
                display_spec.append(("mode_swaps", cparams[0], None))
            elif cs == "unitary":
                nm = params[1].shape[0]
                display_spec.append(("unitary", [cparams[0], cparams[0]+nm-1], 
                                   None))
            elif cs == "group":
                display_spec.append(("group", [params[2], params[3]], 
                                   (params[1])))
            else:
                raise DisplayError("Component in circuit spec not recognised.")
            
        return display_spec
    
    def _freeze_params(self, spec: list|tuple) -> list|tuple:
        """
        Takes a provided circuit spec and will remove get any Parameter objects
        with their currently set values.
        """
        new_spec = []
        # Loop over spec and either call function again or add the value to the
        # new spec
        for s in spec:
            if isinstance(s, (list, tuple)):
                new_spec.append(self._freeze_params(s))
            elif isinstance(s, Parameter):
                new_spec.append(s.get())
            else:
                new_spec.append(s)
        # If original spec was tuple than convert new spec to tuple
        if isinstance(spec, tuple):
            new_spec = tuple(new_spec)
        return new_spec
            
    def _display_loss_check(self, loss: Any) -> bool:
        """
        Checks whether a loss element should be shown when using the display
        function.
        """
        if isinstance(loss, str):
            return True
        elif loss > 0:
            return True
        else:
            return False
        
    def _get_circuit_spec(self) -> list:
        """Returns a copy of the circuit spec attribute."""
        return deepcopy(self.__circuit_spec)