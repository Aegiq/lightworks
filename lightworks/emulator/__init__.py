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
Lightworks Emulator
===================

This module is designed for the simulation of boson sampling in linear optic
photonic circuits. It contains a number of different simulation options, with
each intended for a different particular purpose. The module also contains
capability to simulate a number of imperfections which are typically present
in a boson sampling experiment, such as loss and an imperfect photon source or
detectors.

Simulators:

    Simulator : Directly calculates the probability amplitudes from transitions
        between given inputs and outputs on a circuit.

    Sampler : Calculates the output distribution for a given input state and
        circuit, and enables sampling from it. Imperfect sources and detectors
        can also be utilised here.

    QuickSampler : A faster sampler, which can be used in cases where photon
        number should be preserved and source is perfect.

    Analyzer : Finds all possible outputs of a circuit with a given set of
        inputs, conditional on them meeting a set of post-selection and
        heralding criteria. This means it can be used to analyze how well a
        circuit performs for a given task. It can also produce an error rate
        and value for circuit performance (how often a valid output will be
        produced).

"""

from .backends import Backend
from .components import Detector, Source
from .utils.exceptions import *

__all__ = ["Backend", "Detector", "Source"]
