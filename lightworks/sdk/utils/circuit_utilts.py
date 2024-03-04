"""
Contains a number of different utility functions for modifying circuits.
"""

from .permutation_conversion import permutation_mat_from_swaps_dict
from .permutation_conversion import permutation_to_mode_swaps

import numpy as np

def unpack_circuit_spec(circuit_spec: list) -> list:
    """
    Unpacks and removes any grouped components from a circuit.
    
    Args:
    
        circuit_spec (list) : The circuit spec to unpack.
        
    Returns:
    
        list : The processed circuit spec.
        
    """
    new_spec = [c for c in circuit_spec]
    components = [i[0] for i in circuit_spec]
    while "group" in components:
        temp_spec = []
        for spec in circuit_spec:
            if spec[0] != "group":
                temp_spec += [spec]
            else:
                temp_spec += spec[1][0]
        new_spec = temp_spec
        components = [i[0] for i in new_spec]
    
    return new_spec

def convert_mode_swaps(spec: list, n_modes : int, 
                       preserve_phase: bool = False) -> list:
    """
    Takes a given circuit spec and removes all mode swaps, replacing them
    with fully transmissive beam splitters.
    
    Args:
    
        spec (list) : The circuit spec to remove mode swaps from.
        
        n_modes (int) : The number of modes used in the circuit. This is 
            required to ensure the resulting permutation transform is of the 
            correct size. 
    
        preserve_phase (bool, optional) : Can be used to include additional
            phase shifters to ensure the unitary of each beam splitter is equal 
            to that produced by the mode swap components. Defaults to False.
                            
    Returns:
    
        list : The processed circuit spec.
                                            
    """
    new_spec = []
    for s in spec:
        if s[0] == "mode_swaps":
            U = permutation_mat_from_swaps_dict(s[1][0], n_modes)
            swaps = permutation_to_mode_swaps(U)
            for sw in swaps:
                if preserve_phase:
                    new_spec.append(["ps", (sw, 3*np.pi/2, 0)])
                    new_spec.append(["ps", (sw+1, 3*np.pi/2, 0)])
                new_spec.append(["bs", (sw, sw+1, 0, "Rx")])
        elif s[0] == "group":
            new_s1 = [si for si in s[1]]
            new_s1[0] = convert_mode_swaps(s[1][0], n_modes, preserve_phase)
            s = [s[0], tuple(new_s1)]
            new_spec.append(s)
        else:
            new_spec.append(s)
    return new_spec

def convert_non_adj_beamsplitters(spec: list) -> list:
    """
    Takes a given circuit spec and removes all beam splitters acting on 
    non-adjacent modes by replacing with a mode swap and adjacent beam 
    splitters.
    
    Args:
    
        spec (list) : The circuit spec to remove beam splitter on non-adjacent 
            modes from.
                        
    Returns:
    
        list : The processed circuit spec.
                                            
    """
    new_spec = []
    for s in spec:
        if s[0] == "bs" and abs(s[1][0]-s[1][1]) != 1:
            m1, m2 = s[1][0:2]
            if m1 > m2: m1, m2 = m2, m1
            mid = int((m1+m2-1)/2)
            swaps = {}
            for i in range(m1, mid+1):
                swaps[i] = mid if i == m1 else i - 1
            for i in range(mid+1, m2+1):
                swaps[i] = mid+1 if i == m2 else i + 1
            new_spec.append(["mode_swaps", (swaps, None)])
            # If original modes were inverted then invert here too
            add1, add2 = mid, mid + 1
            if s[1][0] > s[1][1]:
                add1, add2 = add2, add1
            # Add beam splitter on new modes
            new_spec.append(["bs", (add1, add2, s[1][2],s[1][3])])
            swaps = {v : k for k, v in swaps.items()}
            new_spec.append(["mode_swaps", (swaps, None)])
        elif s[0] == "group":
            new_s1 = [si for si in s[1]]
            new_s1[0] = convert_non_adj_beamsplitters(s[1][0])
            s = [s[0], tuple(new_s1)]
            new_spec.append(s)
        else:
            new_spec.append(s)
    return new_spec

def compress_mode_swaps(spec: list) -> list:
    """
    Takes a provided circuit spec and will try to compress any more swaps
    such that the circuit length is reduced. Note that any components in a 
    group will be ignored.
    
    Args:
    
        spec (list) : The circuit spec which is to be processed.
        
    Returns:
    
        list : The processed version of the circuit spec.
        
    """
    new_spec = []
    to_skip = []
    # Loop over each item in original spec
    for i, s in enumerate(spec):
        if i in to_skip:
            continue
        # If it a mode swap then check for subsequent mode swaps
        elif s[0] == "mode_swaps":
            blocked_modes = set()
            for j, s2 in enumerate(spec[i+1:]):
                # Block modes with components other than the mode swap on
                if s2[0] == "ps": 
                    # NOTE: In principle a phase shift doesn't need to 
                    # block a mode and instead we could modify it's 
                    # location
                    blocked_modes.add(s2[1][0])
                elif s2[0] == "bs":
                    blocked_modes.add(s2[1][0])
                    blocked_modes.add(s2[1][1])
                elif s2[0] == "group":
                    for m in range(s2[1][2], s2[1][3]+1):
                        blocked_modes.add(m)
                elif s2[0] == "mode_swaps":
                    # When a mode swap is found check if any of its mode 
                    # are in the blocked mode
                    swaps = s2[1][0]
                    for m in swaps:
                        # If they are then block all other modes of swap
                        if m in blocked_modes:
                            for m in swaps:
                                blocked_modes.add(m)
                            break
                    else:
                        # Otherwise combine the original and found swap
                        # and update spec entry
                        new_swaps = combine_mode_swap_dicts(s[1][0], swaps)
                        s = ["mode_swaps", (new_swaps, None)]
                        # Also set to skip the swap that was combine
                        to_skip.append(i+1+j)
            new_spec.append(s)
        else:
            new_spec.append(s)

    return new_spec
    
def combine_mode_swap_dicts(swaps1: dict, swaps2: dict) -> dict:
    """
    Function to take two mode swap dictionaries and combine them. 
    
    Args:
    
        swaps1 (dict) : The first mode swap dictionary to combine.
        
        swaps2 (dict) : The mode swap dictionary to combine with the first 
            dictionary.
        
    Returns:
    
        dict : The calculated combined mode swap dictionary.
    
    """
    # Store overall swaps in new dictionary
    new_swaps = {}
    added_swaps = []
    for s1 in swaps1:
        for s2 in swaps2:
            # Loop over swaps to combine when a key from swap 2 is in the 
            # values of swap 1
            if swaps1[s1] == s2:
                new_swaps[s1] = swaps2[s2]
                added_swaps.append(s2)
                break
        # If it isn't found then add key and value from swap 1
        else:
            new_swaps[s1] = swaps1[s1]
    # Add any keys from swaps2 that weren't used
    for s2 in swaps2:
        if s2 not in added_swaps:
            new_swaps[s2] = swaps2[s2]
    # Remove any modes that are unchanged as these are not required
    new_swaps = {m1 : m2 for m1, m2 in new_swaps.items() if m1 != m2}
    return new_swaps