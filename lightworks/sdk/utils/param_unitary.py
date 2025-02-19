# Copyright 2025 Aegiq Ltd.
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

from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from lightworks.sdk.circuit.parameters import Parameter


class ParameterizedUnitary:
    """
    Adds support for parametrisation of unitaries for delayed computation with
    sympy.

    Args:

        unitary (sp.Matrix) : The sympy matrix which represents the unitary to
            be implemented.

        params (dict) : A dictionary where the keys should be the names of the
            symbols used in the sympy array and the values are Lightworks
            parameters.

    """

    def __init__(
        self, unitary: sp.Matrix, params: dict[str, Parameter[Any]]
    ) -> None:
        if not isinstance(unitary, sp.Matrix):
            raise TypeError("Supplied unitary must be a sympy matrix.")
        self._unitary = unitary
        self._params = params

    def __str__(self) -> str:
        return str(self._unitary)

    def __repr__(self) -> str:
        return (
            f"ParameterizedUnitary(matrix={self._unitary}, "
            f"params={self._params})"
        )

    @property
    def unitary(self) -> NDArray[np.complex128]:
        """Returns the current value of the unitary matrix."""
        vals = {
            k: p.get() if isinstance(p, Parameter) else p
            for k, p in self._params.items()
        }
        return np.array(self._unitary.evalf(subs=vals), dtype=complex)

    @property
    def shape(self) -> tuple[int, int]:
        """Returns current shape of the unitary matrix."""
        return self._unitary.shape
