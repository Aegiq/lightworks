from .permanent import Permanent
from .determinant import Determinant
from ...sdk import State

from numpy import ndarray

class Backend:
    """General backend for handling both types of statistics."""
    @staticmethod
    def calculate(U: ndarray, in_state: State, out_state: State,
                  statistic_type: str) -> complex:
        if statistic_type == "bosonic":
            return Permanent.calculate(U, in_state, out_state)
        elif statistic_type == "fermionic":
            return Determinant.calculate(U, in_state, out_state)
        else:
            msg = "statistic_type should be 'bosonic' or 'fermionic'."
            raise ValueError(msg)
    