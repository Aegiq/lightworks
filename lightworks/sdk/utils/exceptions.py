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
Defines all custom exceptions used as part of the SDK section of lightworks.
In general, the approach is to use the built-in exceptions where possible, but
provide custom exceptions where the extra context would be useful, or it may 
cause confusion as to the root of an issue if a generic type is used.
"""

class LightworksError(Exception):
    """
    Generic error from which all other errors are derived.
    """
    pass

class StateError(LightworksError):
    """
    Error relating to issues with a provided State
    """
    pass

class ModeRangeError(LightworksError):
    """
    Error for specific errors arising when a provided mode is outside of the 
    circuit range.
    """
    pass

class CircuitCompilationError(LightworksError):
    """
    For all errors that arise during compilation of a circuit.
    """
    pass

class DisplayError(LightworksError):
    """
    Used when specific errors during the Display methods. 
    """
    pass

class ParameterValueError(LightworksError):
    """
    For errors in the setting of a Parameter value.
    """
    pass
    
class ParameterBoundsError(LightworksError):
    """
    For errors in the setting of Parameter bounds.
    """
    pass

class ParameterDictError(LightworksError):
    """
    Exceptions relating to ParameterDict behaviour.
    """
    pass