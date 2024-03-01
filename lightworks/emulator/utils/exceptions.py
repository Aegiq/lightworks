"""
Contains all custom exceptions created for the emulator
"""

class EmulatorError(Exception):
    """
    Generic error for emulator
    """
    pass

class StateError(EmulatorError):
    """
    Error relating to issues with a provided State
    """
    pass

class AnnotatedStateError(EmulatorError):
    """
    Error relating to issues with a provided AnnotatedState
    """
    pass