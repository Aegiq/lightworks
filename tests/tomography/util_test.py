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
    choi_from_unitary,
    density_from_state,
    process_fidelity,
    state_fidelity,
)
from lightworks.tomography.utils import (
    _check_target_process,
    _get_required_tomo_measurements,
    _get_tomo_measurements,
)

U_H = np.array([[2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]], dtype=complex)
U_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
U_CCNOT = np.identity(8, dtype=complex)
U_CCNOT[6:, :] = U_CCNOT[7:5:-1, :]
# NOTE: Currently an issue with scipy.linalg.sqrtm on certain platforms, which
# causes process fidelity to fail for exact matrices. This can be mitigated by
# applying a hadamard to all qubits at the input for these tests.
H2 = np.kron(U_H, U_H)
H3 = np.kron(H2, U_H)
U_CNOT @= H2
U_CCNOT @= H3


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
        "choi",
        [
            choi_from_unitary(U_H),
            choi_from_unitary(U_CNOT),
            choi_from_unitary(U_CCNOT),
        ],
    )
    def test_process_fidelity(self, choi):
        """
        Validate that fidelity value is always 1 when using two identical
        matrices.
        """
        assert process_fidelity(choi, choi) == pytest.approx(1, 1e-3)

    def test_process_fidelity_dim_mismatch(self):
        """
        Check an error is raised if there is a mismatch in dimensions between
        density matrices.
        """
        with pytest.raises(ValueError, match="dimensions"):
            process_fidelity(random_unitary(3), random_unitary(4))

    def test_basic_density_matrix_calc(self):
        """
        Checks the density matrix of the two qubit state |00> is correct.
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
        Checks the calculated density matrix for a bell state is correct.
        """
        rho = density_from_state([2**-0.5, 0, 0, 2**-0.5])
        assert (
            rho.round(5)
            == np.array(
                [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
            )
        ).all()

    def test_hadamard_choi(self):
        """
        Checks that the calculated choi matrix for the hadamard gate is correct.
        """
        choi = choi_from_unitary([[2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]])
        assert (
            choi.round(5)
            == np.array(
                [
                    [0.5, 0.5, 0.5, -0.5],
                    [0.5, 0.5, 0.5, -0.5],
                    [0.5, 0.5, 0.5, -0.5],
                    [-0.5, -0.5, -0.5, 0.5],
                ]
            )
        ).all()

    @pytest.mark.parametrize("n_qubits", [1, 3])
    def test_number_of_measurements(self, n_qubits):
        """
        Confirms that the number of measurements for a full tomography is
        4^n_qubits.
        """
        assert len(_get_tomo_measurements(n_qubits)) == 4**n_qubits

    @pytest.mark.parametrize("remove", [True, False])
    def test_measurements_remove_trivial(self, remove):
        """
        Checks that the trivial I,I measurement is included when required and
        then removed otherwise
        """
        all_meas = _get_tomo_measurements(2, remove_trivial=remove)
        if not remove:
            assert "I,I" in all_meas
        else:
            assert "I,I" not in all_meas

    @pytest.mark.parametrize("n_qubits", [1, 3])
    def test_number_of_required_measurements(self, n_qubits):
        """
        Confirms that the number of required measurements for a tomography is
        3^n_qubits.
        """
        assert len(_get_required_tomo_measurements(n_qubits)[0]) == 3**n_qubits

    @pytest.mark.parametrize("n_qubits", [1, 3])
    def test_required_measurements_order(self, n_qubits):
        """
        Checks that the required measurements function always returns a sorted
        list for consistency between runs.
        """
        req_meas = _get_required_tomo_measurements(n_qubits)[0]
        assert req_meas == sorted(req_meas)

    def test_check_target_process(self):
        """
        Confirm no error is raised when a valid process matrix is provided to
        _check_target_process.
        """
        _check_target_process(random_unitary(4), 2)

    @pytest.mark.parametrize("value", ["Test", {1: 2, 3: 4}, [[1, 2], [3]]])
    def test_check_target_process_type(self, value):
        """
        Confirms an error is raised when an invalid type is provided to
        _check_target_process.
        """
        with pytest.raises(TypeError):
            _check_target_process(value, 2)

    @pytest.mark.parametrize(
        "value",
        [
            random_unitary(4)[:3, :3],
            random_unitary(4)[:, :3],
            random_unitary(4)[:3, :],
        ],
    )
    def test_check_target_process_dimension(self, value):
        """
        Confirms an error is raised when the target process has incorrect
        "dimension.
        """
        with pytest.raises(ValueError):
            _check_target_process(value, 2)
