# Copyright 2024 Aegiq Ltd.
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

import numpy as np
from scipy.linalg import sqrtm


def state_fidelity(rho: np.ndarray, rho_exp: np.ndarray) -> float:
    """
    Calculates the fidelity of the density matrix for a quantum state against
    the expected density matrix.

    Args:

        rho (np.ndarray) : The density matrix of the quantum state.

        rho_exp (np.ndarray) : The expected density matrix.

    Returns:

        float : The calculated fidelity value.

    """
    rho_exp = np.array(rho_exp)
    rho_root = sqrtm(np.array(rho))
    if rho_exp.shape != rho_root.shape:
        msg = (
            "Mismatch in dimensions between provided density matrices, "
            f"{rho_root.shape} & {rho_root.shape}."
        )
        raise ValueError(msg)
    inner = rho_root @ rho_exp @ rho_root
    return abs(np.trace(sqrtm(inner)))


def process_fidelity(chi: np.ndarray, chi_exp: np.ndarray) -> float:
    """
    Desc
    """
    chi * chi_exp
    raise NotImplementedError("Fidelity not yet implemented.")
    return 0
