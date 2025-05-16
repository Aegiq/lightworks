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

from random import shuffle

import numpy as np
import pytest

from lightworks import PostSelection, Sampler, emulator, qubit
from lightworks.tomography import (
    GateFidelity,
    LIProcessTomography,
    MLEProcessTomography,
    choi_from_unitary,
)
from lightworks.tomography.experiments import (
    ProcessTomographyExperiment,
    ProcessTomographyList,
)
from lightworks.tomography.process_tomography import _ProcessTomography


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


cnot_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

h_exp = choi_from_unitary([[2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]])

cnot_exp = choi_from_unitary(cnot_mat)


class TestLIProcessTomography:
    """
    Unit tests for LIProcessTomography routine. This is also used for more
    general testing in instances where datasets are required.
    """

    def setup_class(self):
        """
        Runs process tomography experiments so results can be reused.
        """
        # Hadamard tomography
        n_qubits = 1
        circ = qubit.H()
        experiments = LIProcessTomography(n_qubits, circ).get_experiments()
        self.h_data = run_experiments(experiments, n_qubits)
        # CNOT tomography
        n_qubits = 2
        circ = qubit.CNOT()
        experiments = LIProcessTomography(n_qubits, circ).get_experiments()
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
        tomo = LIProcessTomography(1, qubit.H())
        tomo.process(self.h_data)
        assert tomo.choi == pytest.approx(h_exp, abs=5e-2)

    def test_hadamard_fidelity(self):
        """
        Checks fidelity of hadamard gate process matrix is close to 1.
        """
        tomo = LIProcessTomography(1, qubit.H())
        tomo.process(self.h_data)
        assert tomo.fidelity(h_exp) == pytest.approx(1, 1e-2)

    def test_cnot_choi(self):
        """
        Checks process tomography of the CNOT gate produces the expected choi
        matrix and the fidelity is calculated to be 1.
        """
        tomo = LIProcessTomography(2, qubit.CNOT())
        tomo.process(self.cnot_data)
        assert tomo.choi == pytest.approx(cnot_exp, abs=5e-2)

    def test_cnot_fidelity(self):
        """
        Checks fidelity of CNOT gate process matrix is close to 1.
        """
        tomo = LIProcessTomography(2, qubit.CNOT())
        tomo.process(self.cnot_data)
        assert tomo.fidelity(cnot_exp) == pytest.approx(1, 1e-2)

    def test_cnot_state_fidelity(self):
        """
        Checks that the fidelity of the density matrices for each input is close
        to 1.
        """
        tomo = LIProcessTomography(2, qubit.CNOT())
        fidelity = tomo.process_state_fidelities(self.cnot_data, cnot_mat)
        for f in fidelity.values():
            assert f == pytest.approx(1, 1e-3)

    def test_cnot_density_matrices(self):
        """
        Checks that the calculated density matrices for each input is close to
        the expected input.
        """
        tomo = LIProcessTomography(2, qubit.CNOT())
        rhos = tomo.process_density_matrices(self.cnot_data)
        rhos_exp = tomo.get_expected_density_matrices(cnot_mat)
        for i in rhos:
            assert rhos[i] == pytest.approx(rhos_exp[i], abs=5e-2)

    def test_dict_processing(self):
        """
        Checks process tomography is successful when the data is provided as a
        dictionary.
        """
        tomo = LIProcessTomography(2, qubit.CNOT())
        tomo.process(self.cnot_data_dict)
        assert tomo.fidelity(cnot_exp) == pytest.approx(1, 1e-2)
        assert tomo.choi == pytest.approx(cnot_exp, abs=5e-2)

    def test_dict_processing_shuffled(self):
        """
        Checks process tomography is successful when the data is provided as a
        dictionary and has been shuffled.
        """
        tomo = LIProcessTomography(2, qubit.CNOT())
        shuffled_data = {}
        keys = list(self.cnot_data_dict.keys())
        shuffle(keys)
        for k in keys:
            shuffled_data[k] = self.cnot_data_dict[k]
        tomo.process(shuffled_data)
        assert tomo.fidelity(cnot_exp) == pytest.approx(1, 1e-2)
        assert tomo.choi == pytest.approx(cnot_exp, abs=5e-2)


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

    def test_cnot_state_fidelity(self):
        """
        Checks that the fidelity of the density matrices for each input is close
        to 1. This is checked for MLE tomography as there is a larger number of
        input states.
        """
        tomo = MLEProcessTomography(2, qubit.CNOT())
        fidelity = tomo.process_state_fidelities(self.cnot_data, cnot_mat)
        for f in fidelity.values():
            assert f == pytest.approx(1, 1e-3)

    def test_cnot_density_matrices(self):
        """
        Checks that the calculated density matrices for each input is close to
        the expected input. This is checked for MLE tomography as there is a
        larger number of input states.
        """
        tomo = MLEProcessTomography(2, qubit.CNOT())
        rhos = tomo.process_density_matrices(self.cnot_data)
        rhos_exp = tomo.get_expected_density_matrices(cnot_mat)
        for i in rhos:
            assert rhos[i] == pytest.approx(rhos_exp[i], abs=5e-2)


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
        experiments = GateFidelity(n_qubits, circ).get_experiments()
        self.h_data = run_experiments(experiments, n_qubits)
        # CNOT fidelity
        n_qubits = 2
        circ = qubit.CNOT()
        experiments = GateFidelity(n_qubits, circ).get_experiments()
        self.cnot_data = run_experiments(experiments, n_qubits)

    def test_hadamard_fidelity(self):
        """
        Checks fidelity of hadamard gate process is close to 1.
        """
        tomo = GateFidelity(1, qubit.H())
        f = tomo.process(
            self.h_data, [[2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]]
        )
        assert f == pytest.approx(1, 1e-2)

    def test_cnot_fidelity(self):
        """
        Checks fidelity of CNOT gate process is close to 1.
        """
        tomo = GateFidelity(2, qubit.CNOT())
        f = tomo.process(
            self.cnot_data,
            cnot_mat,
        )
        assert f == pytest.approx(1, 1e-2)


class TestGeneralProcessTomography:
    """
    Contains more general units tests for the process tomography algorithms.
    """

    @pytest.mark.parametrize(
        ("tomography", "exp_len"),
        [
            (LIProcessTomography, 144),
            (MLEProcessTomography, 324),
            (GateFidelity, 144),
        ],
    )
    def test_experiments_length(self, tomography, exp_len):
        """
        Checks number of elements returned by ProcessTomography is the correct
        length.
        """
        tomo = tomography(2, qubit.CNOT())
        experiments = tomo.get_experiments()
        assert len(experiments) == exp_len

    @pytest.mark.parametrize(
        "tomography", [LIProcessTomography, MLEProcessTomography, GateFidelity]
    )
    def test_experiments_list(self, tomography):
        """
        Checks that data returned by ProcessTomography get_experiments is a
        ProcessTomographyList.
        """
        tomo = tomography(2, qubit.CNOT())
        experiments = tomo.get_experiments()
        assert isinstance(experiments, ProcessTomographyList)

    @pytest.mark.parametrize(
        "tomography", [LIProcessTomography, MLEProcessTomography, GateFidelity]
    )
    def test_experiments_in_list_are_experiments(self, tomography):
        """
        Checks that data returned by ProcessTomography get_experiments in a list
        of experiments.
        """
        tomo = tomography(2, qubit.CNOT())
        experiments = tomo.get_experiments()
        for exp in experiments:
            assert isinstance(exp, ProcessTomographyExperiment)

    @pytest.mark.parametrize(
        ("n1", "n2"),
        [
            ("all_circuits", "circuit"),
            ("all_inputs", "input_state"),
            ("all_input_basis", "input_basis"),
            ("all_measurement_basis", "measurement_basis"),
        ],
    )
    def test_all_methods(self, n1, n2):
        """
        Checks that the all methods from ProcessTomographyList correctly returns
        the expected list of values in the correct order.
        """
        tomo = _ProcessTomography(2, qubit.CNOT())
        experiments = tomo.get_experiments()
        for exp, quantity in zip(
            experiments, getattr(experiments, n1), strict=True
        ):
            assert getattr(exp, n2) == quantity
