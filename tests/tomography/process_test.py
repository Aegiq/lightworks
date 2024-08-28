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

from lightworks import PostSelection, emulator, qubit
from lightworks.tomography import ProcessTomography


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
        results.append(sampler.sample_N_outputs(10000, post_select=post_select))
    return results


h_exp = np.zeros((4, 4), dtype=complex)
h_exp[1, 1] = 0.5
h_exp[1, 3] = 0.5
h_exp[3, 1] = 0.5
h_exp[3, 3] = 0.5

chi_exp = np.zeros((16, 16), dtype=complex)
# fmt: off
positive = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1], [12, 0], [12, 1], [0, 12], [1, 12],
    [12, 12], [13, 13]
])
negative = np.array(
    [[13, 0], [13, 1], [0, 13], [1, 13], [12, 13], [13, 12]]
)
# fmt: on
chi_exp[positive[:, 0], positive[:, 1]] = 0.25
chi_exp[negative[:, 0], negative[:, 1]] = -0.25


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

    def test_hadamard_chi(self):
        """
        Checks process tomography of the Hadamard gate produces the expected chi
        matrix.
        """
        assert self.h_tomo.chi == pytest.approx(h_exp, abs=1e-2)

    def test_hadamard_fidelity(self):
        """
        Checks fidelity of hadamard gate process matrix is close to 1.
        """
        assert self.h_tomo.fidelity(h_exp) == pytest.approx(1, 1e-2)

    def test_cnot_chi(self):
        """
        Checks process tomography of the CNOT gate produces the expected chi
        matrix and the fidelity is calculated to be 1.
        """
        assert self.cnot_tomo.chi == pytest.approx(chi_exp, abs=1e-2)

    def test_cnot_fidelity(self):
        """
        Checks fidelity of CNOT gate process matrix is close to 1.
        """
        assert self.cnot_tomo.fidelity(chi_exp) == pytest.approx(1, 1e-2)
