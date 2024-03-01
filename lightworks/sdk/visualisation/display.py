from .draw_circuit_mpl import DrawCircuitMPL
from .draw_circuit_svg import DrawCircuitSVG

import drawsvg
import matplotlib.pyplot as plt

# Display function to interact with relevant classes
def Display(circuit: "Circuit", display_loss: bool = False,                    # type:ignore - Hide warning raised by "Circuit"
            mode_labels: bool = None, display_type: str = "svg"
            ) -> tuple[plt.figure, plt.axes] | drawsvg.Drawing:
    """
    Display
    
    This function can be used to Display a circuit in the quantum emulator as a
    figure in matplotlib.
    
    Args:
    
        circuit (Circuit) : The circuit which is to be displayed.
        
        display_loss (bool, optional) : Choose whether to display loss
            components in the figure, defaults to False.
                                        
        mode_labels (list|None, optional) : Optionally provided a list of mode
            labels which will be used to name the mode something other than 
            numerical values. Can be set to None to use default values.
                                            
        display_type (str, optional) : Selects whether the drawsvg or 
            matplotlib module should be used for displaying the circuit. Should
            either be 'svg' or 'mpl', defaults to 'svg'.
                                           
    Returns:
    
        (fig, ax) | Drawing : The created figure and axis or drawing for the 
            display plot, depending on which display type is used.
        
    Raised:
    
        ValueError : In any cases where an invalid display type is provided an
            exception will be raised.
                                           
    """
    
    if display_type == "mpl":
        disp = DrawCircuitMPL(circuit, display_loss, mode_labels)
        fig, ax = disp.draw()
        return (fig, ax)
    elif display_type == "svg":
        disp = DrawCircuitSVG(circuit, display_loss, mode_labels)
        return disp.draw()
    else:
        raise ValueError("Display type not recognised.")
