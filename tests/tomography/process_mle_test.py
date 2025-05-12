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
from lightworks.tomography import MLEProcessTomography, choi_from_unitary


def run_experiments(experiments, n_qubits):
    """
    Experiment function for testing process tomography. The number of qubits
    should be specified in experiment_args.
    """
    post_select = PostSelection()
    for i in range(n_qubits):
        post_select.add((2 * i, 2 * i + 1), 1)
    results = []
    backend = emulator.Backend("slos")
    for exp in experiments:
        sampler = Sampler(
            exp.circuit,
            exp.input_state,
            20000,
            post_selection=post_select,
            random_seed=99,
        )
        results.append(backend.run(sampler))
    return results


h_exp = choi_from_unitary([[2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]])

cnot_exp = choi_from_unitary(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
)


class TestMLEProcessTomography:
    """
    Unit tests for MLEProcessTomography routine.
    """

    def setup_class(self):
        """
        Runs process tomography experiments so results can be reused.
        """
        # Hadamard tomography
        n_qubits = 1
        circ = qubit.H()
        experiments = MLEProcessTomography(n_qubits, circ).get_experiments()
        self.h_data = run_experiments(experiments, n_qubits)
        # CNOT tomography
        n_qubits = 2
        circ = qubit.CNOT()
        experiments = MLEProcessTomography(n_qubits, circ).get_experiments()
        self.cnot_data = run_experiments(experiments, n_qubits)
        self.cnot_data_dict = {
            (exp.input_basis, exp.measurement_basis): d
            for exp, d in zip(experiments, self.cnot_data, strict=True)
        }

    def test_hadamard_choi(self):
        """
        Checks process tomography of the Hadamard gate produces the expected
        choi matrix.
        """
        tomo = MLEProcessTomography(1, qubit.H())
        tomo.process(self.h_data)
        assert tomo.choi == pytest.approx(h_exp, abs=5e-2)

    def test_hadamard_fidelity(self):
        """
        Checks fidelity of hadamard gate process matrix is close to 1.
        """
        tomo = MLEProcessTomography(1, qubit.H())
        tomo.process(self.h_data)
        assert tomo.fidelity(h_exp) == pytest.approx(1, 1e-2)

    def test_cnot_choi(self):
        """
        Checks process tomography of the CNOT gate produces the expected choi
        matrix and the fidelity is calculated to be 1.
        """
        tomo = MLEProcessTomography(2, qubit.CNOT())
        tomo.process(self.cnot_data)
        assert tomo.choi == pytest.approx(cnot_exp, abs=5e-2)

    def test_cnot_fidelity(self):
        """
        Checks fidelity of CNOT gate process matrix is close to 1.
        """
        tomo = MLEProcessTomography(2, qubit.CNOT())
        tomo.process(self.cnot_data)
        assert tomo.fidelity(cnot_exp) == pytest.approx(1, 1e-2)

    def test_dict_processing(self):
        """
        Checks process tomography is successful when the data is provided as a
        dictionary.
        """
        tomo = MLEProcessTomography(2, qubit.CNOT())
        tomo.process(self.cnot_data_dict)
        assert tomo.fidelity(cnot_exp) == pytest.approx(1, 1e-2)
        assert tomo.choi == pytest.approx(cnot_exp, abs=5e-2)
