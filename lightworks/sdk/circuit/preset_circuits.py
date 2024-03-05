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
Contains a number of pre-defined parametrised circuits which can be used to 
quickly setup different systems and modify them as required. 
"""

from .circuit import Circuit
from .parameters import ParameterDict
from .parameters import Parameter

import numpy as np


class PresetCircuits:
    """
    Contains a selection of pre-defined parameterized circuits which can be
    used to define problems within the Aegiq SDK.
    """
    
    @staticmethod
    def RectangularInterferometer(n_modes: int, length: int = None, 
                                  include_loss_elements: bool = False,
                                  parametrize_loss : bool = False,
                                  default_loss_value: float = 0,
                                  bound_parameters: bool = True
                                  ) -> tuple[Circuit, ParameterDict]:
        """
        Creates a parameterized rectangular interferometer across the set 
        number of modes with the number of columns of elements equal to the 
        value specified in the length option. If being used for simulation then
        lossy elements can also be included.
        
        Args:
        
            n_modes (int) : Set the number of modes that the interferometer 
                will span.
            
            length (int | None, optional) : Sets the number of columns of 
                elements that will be used in the interferometer. If not 
                assigned this will be default to being equal to the number of 
                modes.
            
            include_loss_elements (bool, optional) : Option to include 
                parameterized loss elements after each unit cell in the 
                circuit. Defaults to False.
                                                     
            parametrize_loss (bool, optional) : Select whether or not the loss
                elements should be parameterized. This should usually not be 
                the case when performing optimisations. Defaults to False.
            
            default_loss_value (float, optional) : Sets the default loss value
                to be used when including loss elements with the circuit.
            
            bound_parameters (bool, optional) : Controls whether or not the 
                parameters created will be assigned bounds of 0 - 2*pi. This 
                defaults to True.
                                                
        Returns:
        
            Circuit : The created interferometer circuit.
                                                    
            ParameterDict : A dictionary of all parameters associated with the 
                circuit.
         
        """
        # Default to assume number of modes and length is the same    
        if length is None:
            length = n_modes
        elif not isinstance(length, int):
            if int(length) == length:
                length = int(length)
            else:
                raise TypeError("Length should be an integer value.")
        # Set parameter bounds if these are requested
        bounds = [0, 2*np.pi] if bound_parameters else None
        # Build circuit
        interferometer = Circuit(n_modes)
        parameters = ParameterDict()
        for i in range(length):
            for j in range(i%2, n_modes-1, 2):
                coord = f"{i}_{j}"
                # Create parameters
                parameters["bs_" + coord] = Parameter(0, label = "bs_" + coord,
                                                      bounds = bounds)
                parameters["ps_" + coord] = Parameter(0, label = "ps_" + coord,
                                                      bounds = bounds)
                # Add elements
                interferometer.add_ps(j, phi = parameters["ps_" + coord])
                interferometer.add_bs(j)
                interferometer.add_ps(j+1, phi = parameters["bs_" + coord])
                interferometer.add_bs(j)
                if include_loss_elements:
                    # Add loss elements if required
                    if parametrize_loss:
                        parameters["loss1_"+coord] = Parameter(
                                                      default_loss_value, 
                                                      label = "loss1_" + coord,
                                                      bounds = [0, None])
                        parameters["loss2_"+coord] = Parameter(
                                                      default_loss_value, 
                                                      label = "loss2_" + coord,
                                                      bounds = [0, None])
                        loss1 = parameters["loss1_" + coord]
                        loss2 = parameters["loss2_" + coord]
                    else:
                        loss1 = loss2 = default_loss_value
                    interferometer.add_loss(j, loss = loss1)
                    interferometer.add_loss(j+1, loss = loss2)
            interferometer.add_separation()
        
        return interferometer, parameters
    
    @staticmethod
    def TriangularInterferometer(n_modes: int, n_diagonals: int = None, 
                                 include_loss_elements: bool = False,
                                 parametrize_loss : bool = False,
                                 default_loss_value: float = 0,
                                 bound_parameters: bool = True
                                 ) -> tuple[Circuit, ParameterDict]:
        """
        Creates a parameterized triangular interferometer across the set number
        of modes with the number of diagonal columns being customisable. If 
        being used for simulation then lossy elements can also be included.
        
        Args:
        
            n_modes (int) : Set the number of modes that the interferometer 
                will span.
            
            n_diagonals (int | None, optional) : Sets the number of diagonal 
                columns used to create the interferometer. If not assigned this
                will default to the maximum value of the number of modes - 1.
            
            include_loss_elements (bool, optional) : Option to include 
                parameterized loss elements after each unit cell in the 
                circuit. Defaults to False.
                                                     
            parametrize_loss (bool, optional) : Select whether or not the loss
                elements should be parameterized. Defaults to False.
            
            default_loss_value (float, optional) : Sets the default loss value
                to be used when including loss elements with the circuit.
            
            bound_parameters (bool, optional) : Controls whether or not the 
                parameters created will be assigned bounds of 0 - 2*pi. This 
                defaults to True.
                                                
        Returns:
        
            Circuit : The created interferometer circuit.
                                                    
            ParameterDict : A dictionary of all parameters associated with the 
                circuit.
         
        """
        # Default to assume number of modes and length is the same    
        if n_diagonals is None:
            n_diagonals = n_modes - 1
        elif not isinstance(n_diagonals, int):
            if int(n_diagonals) == n_diagonals:
                n_diagonals = int(n_diagonals)
            else:
                raise TypeError("n_diagonals should be an integer value.")
            if n_diagonals > n_modes - 1:
                raise ValueError("n_diagonals should be <= n_modes - 1.")
        # Set parameter bounds if these are requested
        bounds = [0, 2*np.pi] if bound_parameters else None
        # Build circuit
        interferometer = Circuit(n_modes)
        parameters = ParameterDict()
        for i in range(n_diagonals):
            for j in range(0, n_modes-1-i, 1):
                interferometer.add_separation([j,j+1])
                coord = f"{j+2*i}_{j}"
                # Create parameters
                parameters["bs_" + coord] = Parameter(0, label = "bs_" + coord,
                                                      bounds = bounds)
                parameters["ps_" + coord] = Parameter(0, label = "ps_" + coord,
                                                      bounds = bounds)
                # Add elements
                interferometer.add_ps(j, phi = parameters["ps_" + coord])
                interferometer.add_bs(j)
                interferometer.add_ps(j+1, phi = parameters["bs_" + coord])
                interferometer.add_bs(j)
                if include_loss_elements:
                    # Add loss elements if required
                    if parametrize_loss:
                        parameters["loss1_"+coord] = Parameter(
                                                      default_loss_value, 
                                                      label = "loss1_" + coord,
                                                      bounds = [0, None])
                        parameters["loss2_"+coord] = Parameter(
                                                      default_loss_value, 
                                                      label = "loss2_" + coord,
                                                      bounds = [0, None])
                        loss1 = parameters["loss1_" + coord]
                        loss2 = parameters["loss2_" + coord]
                    else:
                        loss1 = loss2 = default_loss_value
                    interferometer.add_loss(j, loss = loss1)
                    interferometer.add_loss(j+1, loss = loss2)
        
        return interferometer, parameters