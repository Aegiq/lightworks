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
Contains all custom exceptions created for the emulator section of Lightworks.
"""

from ...sdk.utils import LightworksError


class EmulatorError(LightworksError):
    """
    Generic error for emulator which sub classes the main LightworksError
    """


class AnnotatedStateError(EmulatorError):
    """
    Error relating to issues with a provided AnnotatedState
    """


class ResultCreationError(EmulatorError):
    """
    For specific errors which occur when using creating a Result object.
    """


class BackendError(EmulatorError):
    """
    Raised when errors occur in the Backend object.
    """


class ModeMismatchError(EmulatorError):
    """
    For use in simulation objects when there is a mode mismatch between
    provided states/circuit.
    """


class PhotonNumberError(EmulatorError):
    """
    For use in simulation objects when there is a photon number mismatch
    between inputs and/or outputs.
    """


class SamplerError(EmulatorError):
    """
    Specific error in the Sampler objects.
    """
