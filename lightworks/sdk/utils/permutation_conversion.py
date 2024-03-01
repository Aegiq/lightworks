import numpy as np

def permutation_mat_from_swaps_dict(swaps: dict, n_modes: int) -> np.ndarray:
    """
    Calculates the permutation unitary for a given dictionary of swaps across
    the n_modes of a circuit.
    
    Args:
    
        swaps (dict) : The dictionary containing the target mode swaps.
        
        n_modes (int) : The number of modes in the circuit. If this is not the
            number of modes then an incorrect dimension unitary will be 
            returned.
                        
    Returns:
    
        np.ndarray : The determined permutation matrix for the provided set of 
            mode swaps.
            
    """
    if not isinstance(swaps, dict):
        raise TypeError("swaps should be a dictionary object.")
    
    # Add in missing modes from swap dictionary
    full_swaps = {}
    for m in range(n_modes):
        if m not in swaps:
            full_swaps[m] = m
        else:
            full_swaps[m] = swaps[m]
    # Create swap unitary
    permutation = np.zeros((n_modes, n_modes), dtype=complex)
    for i, j in full_swaps.items():
        permutation[j,i] = 1
    
    return permutation

def permutation_to_mode_swaps(U: np.ndarray) -> list:
    """
    Converts a permutation matrix into an ordered set of modes swaps between
    adjacent modes.
    
    Args:
    
        U (np.ndarray) : The square permutation matrix that is to be decomposed
            into swaps.
    
    Returns:
    
        list : An ordered list of modes that need to be swapped. With the value
            in the list being the lower of the two adjacent modes that are to 
            be swapped.
    
    """
    if not isinstance(U, (np.ndarray, list)):
        msg = "Provided permutation matrix should be numpy array or list."
        raise TypeError(msg)
    U = np.array(U)
    Uc = U.copy()
    N = Uc.shape[0]
    # Check matrix is permutation matrix
    if Uc.shape[0] != Uc.shape[1]:
        raise ValueError("Provided permutation matrix must be square.")
    for i in range(Uc.shape[0]):
        for j in range(Uc.shape[0]):
            if not abs(Uc[i,j])**2 > 1-1e-10 and not abs(Uc[i,j])**2 < 1e-10:
                msg = """Provided matrix is not a permutation matrix, it should 
                         contain only zeros or ones."""
                raise ValueError(" ".join(msg.split()))
    # Determine required mode swaps and store
    s1 = []
    s2 = []
    # Use logic here to start in bottom left corner of matrix and then
    # zig-zag upwards
    for n_col in range(0, N - 1):
        for n in range(n_col + 1):
            if n_col%2 == 1:
                i = N - n_col - 1 + n
                j = n
            else:
                i = N - 1 - n
                j = n_col - n
            # Only apply swap for any non-zero elements
            if Uc[i,j] > 1e-3:
                if n_col%2 == 1:
                    Us = swapmat(N, i-1, i)
                    Uc = Us@Uc
                    s1 += [i-1]
                else:
                    Us = swapmat(N, j, j+1)
                    Uc = Uc@Us
                    s2 += [j]
    # Combine list to create full swaps
    swaps = s2 + list(reversed(s1))
    return swaps
    
def swapmat(N, m1, m2):
    """Determine unitary to perform mode swap between two modes"""
    U = np.identity(N)
    U[m1, m1] = 0
    U[m2, m1] = 1
    U[m1, m2] = 1
    U[m2, m2] = 0
    return U