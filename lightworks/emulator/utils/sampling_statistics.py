def set_statistic_type(new_type: str) -> None:
    """
    Update the sampling statistic type used in the emulator.
    
    Args:
    
        new_type (str) : The new sampling statistics to use, should be either
            bosonic or fermionic.
    
    """
    from ...emulator import __settings
    
    if new_type not in ["bosonic", "fermionic"]:
        msg = "Statistic type should have value of 'bosonic' or 'fermionic'."
        raise ValueError(msg)
    
    __settings["statistic_type"] = new_type
    
def get_statistic_type() -> str:
    """
    Retrieve the current sampling statistic type used by the emulator.
    
    Returns:
    
        str : The current sampling statistics of the emulator.
        
    """
    from ...emulator import __settings
    return __settings["statistic_type"]