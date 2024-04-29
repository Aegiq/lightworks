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
The circuit class enables the creation of circuits from a number of linear 
optics components. It also supports the inclusion of loss when being used as
part of simulations.
"""

from ..utils import ModeRangeError, check_unitary
from ..utils import permutation_mat_from_swaps_dict
from ..visualisation import Display

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from copy import copy


class CompiledCircuit:
    """
    Used for building an arbitrary photonic circuit from a set of fundamental 
    components, creating the unitary matrices which represent each component
    and combining them together. This class is not intended to be directly 
    called, and instead CompiledCircuits should be created through the _build
    method of Circuit if it is required.
    
    Args:
    
        n_modes (int) : The number of modes to use for the circuit.
    
    Attributes:
    
        U (np.ndarray) : The target unitary, as defined by the components
            added to the circuit.
        
        U_full (np.ndarray) : The calculated effective unitary, if loss is 
            included then this will have a larger dimension than the provided 
            unitary as some coupling to imaginary loss modes will occur.
        
        _loss_modes (int) : The number of additional loss modes included to
            simulate loss. This should not be modified.
    
    """
    def __init__(self, n_modes: int) -> None:
        
        if not isinstance(n_modes, int):
            if int(n_modes) == n_modes:
                n_modes = int(n_modes)
            else:
                raise TypeError("Number of modes should be an integer.")
        self._n_modes = n_modes
        self._loss_included = False
        self.U = np.identity(n_modes, dtype = complex)
        self.U_full = np.identity(n_modes, dtype = complex)
        self._loss_modes = 0
        self._in_heralds = {}
        self._out_heralds = {}
        
        return
    
    def __add__(self, value: "CompiledCircuit") -> "CompiledCircuit":
        """Method for addition of two circuits."""
        if isinstance(value, CompiledCircuit):
            if self.n_modes != value.n_modes:
                raise ModeRangeError(
                    "Mismatch in number of circuit modes, used add method to "
                    "add circuits of a different size.")
            nm = self.n_modes
            newU = value.U @ self.U 
            loss = self._loss_included or value._loss_included
            if loss:
                l1 = self._loss_modes
                l2 = value._loss_modes
                newU_full = np.identity(nm + l1 + l2, dtype=complex)
                # Expand U full to include combined loss sum
                newU_full[:nm+l1, :nm+l1] = self.U_full
                U2_full = np.identity(nm + l1 + l2, dtype = complex)
                U2_full[:nm, :nm] = value.U_full[:nm, :nm]
                U2_full[nm+l1:, :nm] = value.U_full[nm:, :nm]
                U2_full[:nm, nm+l1:] = value.U_full[:nm, nm:]
                U2_full[nm+l1:, nm+l1:] = value.U_full[nm:, nm:]
                newU_full = U2_full @ newU_full
            else:
                newU_full = newU
                l1, l2 = 0, 0
            # Create a new circuit and manually assign attributes
            newCirc = CompiledCircuit(nm)
            newCirc._loss_included = loss
            newCirc.U = newU
            newCirc.U_full = newU_full
            newCirc._loss_modes = l1 + l2
            return newCirc
        else:
            raise TypeError("Addition only supported between two circuits.")
        
    @property
    def n_modes(self) -> int:
        return self._n_modes

    @n_modes.setter
    def n_modes(self, value: Any) -> None:
        raise AttributeError(
            "Number of modes should not be modified after Circuit creation.")
        
    @property
    def loss_modes(self):
        """Returns number of loss modes used in the circuit."""
        return self._loss_modes
    
    @property
    def heralds(self) -> dict:
        return {"input" : copy(self._in_heralds), 
                "output" : copy(self._out_heralds)}
        
    def add(self, circuit: "CompiledCircuit", mode: int) -> None:
        """
        Add a circuit which is of a different size (but smaller than) the 
        original circuit.
        
        Args:
        
            circuit (Circuit) : The smaller circuit which is to be added onto
                the existing circuit.
                                
            mode (int) : The mode which the circuit is being added should start
                from. Note that this shouldn't cause the new circuit components
                to exceed the existing circuit size.
                         
            group (bool, optional) : Controls whether all components of the 
                circuit being added should be displayed or if it can all be 
                combined into a single element.
                                     
            name (str, optional) : Set a name to use when displaying the 
                circuit element. Be aware that longer names may not be 
                compatible.   
        
        """
        # Check provided entries are valid
        self._mode_in_range(mode)
        if not isinstance(circuit, (CompiledCircuit, CompiledUnitary)): 
            raise TypeError(
                "Addition only supported between two compiled circuits.")
        if mode + circuit.n_modes > self.n_modes:
            raise ModeRangeError("Circuit to add is outside of mode range")
        
        nm = circuit.n_modes
        # Convert normal unitary to larger size
        Uc = circuit.U
        new_U = np.identity(self.n_modes, dtype=complex)
        new_U[mode:mode+nm, mode:mode+nm] = Uc[:,:]
        # Then convert full unitary
        Uc_full = circuit.U_full
        if circuit._loss_included:
            lm = circuit._loss_modes
            new_U_full = np.identity(self.n_modes+lm, dtype=complex)
            new_U_full[mode:mode+nm, mode:mode+nm] = Uc_full[:nm, :nm]
            new_U_full[self.n_modes:, self.n_modes:] = Uc_full[nm:, nm:]
            new_U_full[mode:mode+nm, self.n_modes:] = Uc_full[:nm, nm:]
            new_U_full[self.n_modes:, mode:mode+nm] = Uc_full[nm:, :nm]
        else:
            new_U_full = new_U.copy()
        # Assign modified attributes to circuit to add
        to_add = CompiledCircuit(self.n_modes)
        to_add.U = new_U
        to_add.U_full = new_U_full
        to_add._loss_included = circuit._loss_included
        to_add._loss_modes = circuit._loss_modes
        # Use built in addition function to combine the circuit
        new_circuit = self + to_add
        # Copy created attributes from new circuit
        for k in self.__dict__.keys():
            setattr(self, k, getattr(new_circuit, k))
            
        return
    
    def add_bs(self, mode_1: int, mode_2: int | None = None, 
               reflectivity: float = 0.5, convention: str = "Rx") -> None:
        """
        Function to add a beam splitter of specified reflectivity between two 
        modes to the circuit.
        
        Args:
        
            mode_1 (int) : The first mode which the beam splitter acts on.
            
            mode_2 (int | None, optional) : The second mode that the beam 
                splitter acts on. This can also be left as the default value of
                None to automatically use mode_2 as mode_1 + 1.
            
            reflectivity (float, optional) : The reflectivity value of the beam 
                splitter. Defaults to 0.5.
                                     
            convention (str, optional) : The convention to use for the beam 
                splitter, should be either "Rx" (the default value) or "H".
                
        """
        # Assign mode_2 if not already done
        if mode_2 is None:
            mode_2 = mode_1 + 1
        if mode_1 == mode_2:
            raise ValueError(
                "Beam splitter must act across two different modes.")
        # Check correct convention is given
        all_convs = ["Rx", "H"]
        if convention not in all_convs:
            msg = "Provided beam splitter convention should in ['"
            for conv in all_convs:
                msg += conv + ", "
            msg = msg[:-2] + "']"
            raise ValueError(msg)
        # Check valid beam splitter reflectivity
        if reflectivity < 0 or reflectivity > 1:
            raise ValueError(
                "Beam splitter reflectivity should be in range [0,1].")
        self._mode_in_range(mode_1)
        self._mode_in_range(mode_2)
        # Find unitary assuming no loss     
        U_bs = np.identity(self.n_modes, dtype=complex)
        U_bs_full = np.identity(self.n_modes+self.loss_modes, dtype=complex)
        theta = np.arccos(reflectivity**0.5)
        for arr in [U_bs, U_bs_full]:
            if convention == "Rx":
                arr[mode_1, mode_1] = np.cos(theta)
                arr[mode_1, mode_2] = 1j*np.sin(theta)
                arr[mode_2, mode_1] = 1j*np.sin(theta)
                arr[mode_2, mode_2] = np.cos(theta)
            elif convention == "H":
                arr[mode_1, mode_1] = np.cos(theta)
                arr[mode_1, mode_2] = np.sin(theta)
                arr[mode_2, mode_1] = np.sin(theta)
                arr[mode_2, mode_2] = -np.cos(theta)
        # Update unitaries
        self.U = U_bs @ self.U
        self.U_full = U_bs_full @ self.U_full
        return
    
    def add_ps(self, mode: int, phi: float) -> None:
        """
        Applies a phase shift to a given mode in the circuit.
        
        Args:
        
            mode (int) : The mode on which the phase shift acts.
            
            phi (float) : The angular phase shift to apply.
                                                          
        """
        self._mode_in_range(mode)
        # Create new unitary
        U_ps = np.identity(self.n_modes, dtype=complex)
        U_ps[mode, mode] = np.exp(1j*phi)
        U_ps_full = np.identity(self.n_modes+self.loss_modes, dtype=complex)
        U_ps_full[mode, mode] = np.exp(1j*phi)
        self.U = U_ps @ self.U
        # Calculate U_full with loss
        self.U_full = U_ps_full @ self.U_full
        return
    
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
        # Create new unitary
        U_lc = np.identity(self.n_modes, dtype=complex)
        self.U = U_lc @ self.U
        # Calculate U_full with loss
        if self._loss_included:
            n = 1 if loss > 0 else 0
            self._loss_modes = self._loss_modes + n
            # Switch U_full and U_ps to larger size
            U_full = self._grow_unitary(self.U_full, n)
            # Create loss matrix and multiply
            mode_total = self.n_modes + self._loss_modes
            if loss > 0:
                U_lc = self._loss_mat(mode_total, mode, mode_total-1, loss)
            else:
                U_lc = np.identity(mode_total, dtype = complex)
        else:
            U_full = self.U_full
        self.U_full = U_lc @ U_full
        return
    
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
        # Check provided data is valid
        if not isinstance(swaps, dict):
            raise TypeError("Mode swaps data should be a dictionary.")
        for m in list(swaps.keys()) + list(swaps.values()):
            self._mode_in_range(m)
        in_modes = sorted(list(swaps.keys()))
        out_modes = sorted(list(swaps.values()))
        if in_modes != out_modes:
            raise ValueError(
                "Mode swaps not complete, dictionary keys and values should "
                "contain the same modes.")
        # Create swap unitary
        U = permutation_mat_from_swaps_dict(swaps, self.n_modes)
        # Expand to include loss modes
        U_full = np.identity(self.n_modes + self._loss_modes, dtype=complex)
        U_full[:self.n_modes, :self.n_modes] = U[:, :]
        # Combine with existing matrices
        self.U = U @ self.U
        self.U_full = U_full @ self.U_full

        return
    
    def add_herald(self, n_photons: int, input_mode: int, 
                   output_mode: int = None) -> None:
        """
        Add a herald across a selected input/output of the circuit. If only one
        mode is specified then this will be used for both the input and output.
        
        Args:
        
            n_photons (int) : The number of photons to use for the heralding.
            
            input_mode (int) : The input mode to use for the herald.
            
            output_mode (int | None, optional) : The output mode for the 
                herald, if this is not specified it will be set to the value of
                the input mode.
        
        """
        if not isinstance(n_photons, int) or isinstance(n_photons, bool):
            raise TypeError(
                "Number of photons for herald should be an integer.")
        n_photons = int(n_photons)
        if output_mode is None:
            output_mode = input_mode
        self._mode_in_range(input_mode)
        self._mode_in_range(output_mode)
        # Check if herald already used on input or output
        if input_mode in self._in_heralds:
            raise ValueError("Heralding already set for chosen input mode.")
        if output_mode in self._out_heralds:
            raise ValueError("Heralding already set for chosen output mode.")
        # If not then update dictionaries
        self._in_heralds[input_mode] = n_photons
        self._out_heralds[output_mode] = n_photons
    
    def _grow_unitary(self, U: np.ndarray, n: int) -> np.ndarray:
        """Function to grow a unitary by a given n modes"""
        ns = U.shape[0]
        Ug = np.identity(ns + n, dtype = complex)
        Ug[:ns, :ns] = U
        return Ug
    
    def _loss_mat(self, n: int, mode_1: int, mode_2: int, 
                  loss: float) -> np.ndarray:
        """Create a unitary loss matrix between two modes"""
        T = 10**(-loss/10)
        U_loss = np.identity(n, dtype = complex)
        U_loss[mode_1, mode_1] = T**0.5
        U_loss[mode_2, mode_2] = T**0.5
        U_loss[mode_1, mode_2] = (1-T)**0.5
        U_loss[mode_2, mode_1] = (1-T)**0.5
        return U_loss
    
    def _mode_in_range(self, mode: int) -> bool:
        """Check a mode exists within the created circuit and also confirm it 
           is an integer."""
        if not isinstance(mode, int):
            if int(mode) == mode:
                mode = int(mode)
            else:
                raise TypeError("Mode number should be an integer.")
        if 0 <= mode < self.n_modes:
            return True
        else:
            raise ModeRangeError(
                "Selected mode(s) is not within the range of the created "
                "circuit.")
        
    def _check_loss(self, loss: float) -> bool:
        """Check that loss is positive and toggle _loss_included if not already
           done."""
        if loss < 0:
            raise ValueError("Provided loss values should be greater than 0.")
        if loss > 0:
            if not self._loss_included:
                self._loss_included = True
                
class CompiledUnitary(CompiledCircuit):
    """
    This class can be used to create a circuit which implements the target 
    provided unitary across all of its modes.
    
    Args:
    
        unitary (np.ndarray) : The target NxN unitary matrix which is to be 
            implemented.
    
    """
    def __init__(self, unitary: np.ndarray, label: str = "U") -> None:
        
        # Check type of supplied unitary
        if not isinstance(unitary, (np.ndarray, list)):
            raise TypeError("Unitary should be a numpy array or list.")
        unitary = np.array(unitary)
        
        # Ensure unitary is valid
        if not check_unitary(unitary):
            raise ValueError("Matrix is not unitary.")
        if not isinstance(label, str):
            raise TypeError("Label for unitary should be a string.")
        
        # Also check label
        # Assign attributes of unitary
        self.U = unitary
        self.U_full = unitary
        self._n_modes = unitary.shape[0]
        self._loss_included = False
        self._loss_modes = 0
        
        return