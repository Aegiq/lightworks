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

from lightworks import PostSelection, Sampler, emulator, qubit
from lightworks.tomography import GateFidelity


def experiment(circuits, inputs, n_qubits):
    """
    Experiment function for testing process tomography. The number of qubits
    should be specified in experiment_args.
    """
    post_select = PostSelection()
    for i in range(n_qubits):
        post_select.add((2 * i, 2 * i + 1), 1)
    results = []
    backend = emulator.Backend("slos")
    for circ, in_s in zip(circuits, inputs, strict=True):
        sampler = Sampler(
            circ, in_s, 20000, post_selection=post_select, random_seed=99
        )
        results.append(backend.run(sampler))
    return results


class TestGateFidelity:
    """
    Unit tests for checking GateFidelity routine.
    """

    def setup_class(self):
        """
        Runs experiments so results can be reused.
        """
        # Hadamard fidelity
        n_qubits = 1
        circ = qubit.H()
        self.h_tomo = GateFidelity(n_qubits, circ, experiment, [n_qubits])
        self.h_tomo.process([[2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]])
        # CNOT fidelity
        n_qubits = 2
        circ = qubit.CNOT()
        cnot_tomo = GateFidelity(n_qubits, circ, experiment, [n_qubits])
        cnot_tomo.process(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        )
        self.cnot_tomo = cnot_tomo

    def test_hadamard_fidelity(self):
        """
        Checks fidelity of hadamard gate process is close to 1.
        """
        assert self.h_tomo.fidelity == pytest.approx(1, 1e-2)

    def test_cnot_fidelity(self):
        """
        Checks fidelity of CNOT gate process is close to 1.
        """
        assert self.cnot_tomo.fidelity == pytest.approx(1, 1e-2)
