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

import pytest

from lightworks import PostSelection, emulator, qubit
from lightworks.tomography import ProcessTomography, choi_from_unitary


def experiment(circuits, inputs, n_qubits):
    """
    Experiment function for testing process tomography. The number of qubits
    should be specified in experiment_args.
    """
    post_select = PostSelection()
    for i in range(n_qubits):
        post_select.add((2 * i, 2 * i + 1), 1)
    results = []
    for circ, in_s in zip(circuits, inputs):
        sampler = emulator.Sampler(circ, in_s, backend="slos")
        results.append(
            sampler.sample_N_outputs(20000, post_select=post_select, seed=99)
        )
    return results


h_exp = choi_from_unitary([[2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]])

cnot_exp = choi_from_unitary(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
)


class TestProcessTomography:
    """
    Unit tests for ProcessTomography routine.
    """

    def setup_class(self):
        """
        Runs process tomography experiments so results can be reused
        """
        # Hadamard tomography
        n_qubits = 1
        circ = qubit.H()
        self.h_tomo = ProcessTomography(n_qubits, circ, experiment, [n_qubits])
        self.h_tomo.process()
        # CNOT tomography
        n_qubits = 2
        circ = qubit.CNOT()
        self.cnot_tomo = ProcessTomography(
            n_qubits, circ, experiment, [n_qubits]
        )
        self.cnot_tomo.process()

    def test_hadamard_choi(self):
        """
        Checks process tomography of the Hadamard gate produces the expected
        choi matrix.
        """
        assert self.h_tomo.choi == pytest.approx(h_exp, abs=5e-2)

    def test_hadamard_fidelity(self):
        """
        Checks fidelity of hadamard gate process matrix is close to 1.
        """
        assert self.h_tomo.fidelity(h_exp) == pytest.approx(1, 1e-2)

    def test_cnot_choi(self):
        """
        Checks process tomography of the CNOT gate produces the expected choi
        matrix and the fidelity is calculated to be 1.
        """
        assert self.cnot_tomo.choi == pytest.approx(cnot_exp, abs=5e-2)

    def test_cnot_fidelity(self):
        """
        Checks fidelity of CNOT gate process matrix is close to 1.
        """
        assert self.cnot_tomo.fidelity(cnot_exp) == pytest.approx(1, 1e-2)
