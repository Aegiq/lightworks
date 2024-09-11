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
import pytest

from lightworks import random_unitary
from lightworks.tomography import (
    density_from_state,
    process_fidelity,
    state_fidelity,
)


class TestUtils:
    """
    Unit tests for utilities of the tomography module.
    """

    @pytest.mark.parametrize(
        "rho",
        [
            [[1, 0], [0, 0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5j], [-0.5j, 0.5]],
        ],
    )
    def test_state_fidelity(self, rho):
        """
        Validate that fidelity value is always 1 when using two identical
        matrices.
        """
        rho = np.array(rho, dtype=complex)
        assert state_fidelity(rho, rho) == pytest.approx(1, 1e-6)

    def test_state_fidelity_dim_mismatch(self):
        """
        Check an error is raised if there is a mismatch in dimensions between
        density matrices.
        """
        with pytest.raises(ValueError, match="dimensions"):
            state_fidelity(random_unitary(3), random_unitary(4))

    @pytest.mark.parametrize(
        "rho",
        [
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0.5, 0, 0.5], [0, 0, 0, 0], [0, 0.5, 0, 0.5]],
            [[0, 0, 0, 0], [0, 0.5, 0, 0.5j], [0, 0, 0, 0], [0, -0.5j, 0, 0.5]],
        ],
    )
    def test_process_fidelity(self, rho):
        """
        Validate that fidelity value is always 1 when using two identical
        matrices.
        """
        rho = np.array(rho, dtype=complex)
        assert process_fidelity(rho, rho) == pytest.approx(1, 1e-6)

    def test_process_fidelity_dim_mismatch(self):
        """
        Check an error is raised if there is a mismatch in dimensions between
        density matrices.
        """
        with pytest.raises(ValueError, match="dimensions"):
            process_fidelity(random_unitary(3), random_unitary(4))

    def test_basic_density_matrix_calc(self):
        """
        Desc
        """
        rho = density_from_state([1, 0, 0, 0])
        assert (
            rho
            == np.array(
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            )
        ).all()

    def test_bell_state_density_matrix_calc(self):
        """
        desc
        """
        rho = density_from_state([2**-0.5, 0, 0, 2**-0.5])
        assert (
            rho.round(5)
            == np.array(
                [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
            )
        ).all()
