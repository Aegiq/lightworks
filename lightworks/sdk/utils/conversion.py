"""
This file contains a number of functions for converting between quantities in 
the emulator.
"""

from math import log10

def db_loss_to_transmission(loss: float) -> float:
    """
    Function to convert from a given dB loss into the equivalent transmission 
    value as a percentage. Note this function does not support conversion of 
    gain values.
    
    Args:
    
        loss (float) : The loss value in decibels. 
    
    Returns:
    
        float : The calculated transmission as a decimal.
    
    """
    # Standardize loss format
    loss = -abs(loss)
    return 10**(loss/10)

def transmission_to_db_loss(transmission: float) -> float:
    """
    Function to convert from a transmission into dB loss. This dB loss will be
    returned as a positive value.
    
    Args:
    
        transmission (float) : The transmission value as a decimal, this should
            be in the range (0,1].
        
    Returns:
    
        float : The calculated dB loss. This is returned as a positive value.
        
    Raises:
    
        ValueError: Raised in cases where transmission is not in range (0,1].
    
    """
    if transmission <= 0 or transmission > 1:
        raise ValueError("Transmission value should be in range (0,1].")
    return abs(10*log10(transmission))