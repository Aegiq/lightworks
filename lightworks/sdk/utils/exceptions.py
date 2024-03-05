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
Contains all custom exceptions created for the emulator
"""

class SDKError(Exception):
    """
    Generic error for SDK
    """
    pass

class StateError(SDKError):
    """
    Error relating to issues with a provided State
    """
    pass

class CircuitError(SDKError):
    """
    For all errors related to circuits.
    """
    pass

class ModeRangeError(CircuitError):
    """
    Error for specific errors arising when a provided mode is outside of the 
    circuit range.
    """
    pass

class CircuitCompilationError(SDKError):
    """
    For all errors that arise during compilation of a circuit.
    """
    pass

class CompiledCircuitError(SDKError):
    """
    For errors arising in CompiledCircuit class which do not fall under typical
    python exceptions.
    """
