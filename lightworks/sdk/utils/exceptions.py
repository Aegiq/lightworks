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
