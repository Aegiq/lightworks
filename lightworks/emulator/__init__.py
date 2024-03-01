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

from .components import *
from .simulation import *
from .utils import *

__all__ = ["Simulator", "Sampler", "fidelity", "Source", "Detector", 
           "Analyzer", "EmulatorError", "StateError", "QuickSampler", 
           "set_statistic_type"]

# Store type of sampling statistics to use
__settings = {}
__settings["statistic_type"] = "bosonic"