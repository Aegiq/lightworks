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

from lightworks import Circuit, PostSelection, State, qubit
from lightworks.emulator import Sampler
from lightworks.tomography import StateTomography


def experiment(circuits):
    """
    Experiment function for testing state tomography functionality is correct.
    """
    # Find number of qubits using available input modes.
    n_qubits = int(circuits[0].input_modes / 2)
    n_samples = 25000
    post_selection = PostSelection()
    for i in range(n_qubits):
        post_selection.add((2 * i, 2 * i + 1), 1)
    results = []
    for circ in circuits:
        sampler = Sampler(circ, State([1, 0] * n_qubits), backend="slos")
        results.append(
            sampler.sample_N_outputs(n_samples, post_select=post_selection)
        )
    return results


class TestStateTomography:
    """
    Unit tests for state tomography class.
    """

    @pytest.mark.flaky(max_runs=3)
    @pytest.mark.parametrize("n_qubits", [1, 2])
    def test_basic_state(self, n_qubits):
        """
        Checks correct density matrix is produced when performing tomography on
        the |0> X n_qubits state.
        """
        base_circ = Circuit(n_qubits * 2)
        tomo = StateTomography(n_qubits, base_circ, experiment)
        rho = tomo.process()
        rho_exp = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        rho_exp[0, 0] = 1
        assert rho == pytest.approx(rho_exp, abs=1e-2)
        assert tomo.fidelity(rho_exp) == pytest.approx(1, 1e-3)

    @pytest.mark.flaky(max_runs=3)
    @pytest.mark.parametrize("n_qubits", [1, 2])
    def test_ghz_state(self, n_qubits):
        """
        Checks correct density matrix is produced when performing tomography on
        the n_qubit GHZ state.
        """
        base_circ = Circuit(n_qubits * 2)
        base_circ.add(qubit.H())
        for i in range(n_qubits - 1):
            base_circ.add(qubit.CNOT(), 2 * i)
        tomo = StateTomography(n_qubits, base_circ, experiment)
        rho = tomo.process()
        rho_exp = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        rho_exp[0, 0] = 0.5
        rho_exp[0, -1] = 0.5
        rho_exp[-1, 0] = 0.5
        rho_exp[-1, -1] = 0.5
        assert rho == pytest.approx(rho_exp, abs=1e-2)
        assert tomo.fidelity(rho_exp) == pytest.approx(1, 1e-3)

    @pytest.mark.parametrize("n_modes", [2, 3, 5])
    def test_number_of_input_modes_twice_number_of_qubits(self, n_modes):
        """
        Checks that number of input modes must be twice number of qubits,
        corresponding to dual rail encoding.
        """
        with pytest.raises(ValueError):
            StateTomography(2, Circuit(n_modes), experiment)

    @pytest.mark.parametrize("value", [Circuit(4), 4, None])
    def test_experiment_must_be_function(self, value):
        """
        Checks value of experiment must be a function.
        """
        with pytest.raises(TypeError):
            State(2, Circuit(4), value)

    def test_density_mat_before_calc(self):
        """
        Checks an error is raised
        """
        tomo = StateTomography(2, Circuit(4), experiment)
        with pytest.raises(AttributeError):
            tomo.rho  # noqa: B018

    def test_base_circuit_unmodified(self):
        """
        Confirms base circuit is unmodified when performing single qubit
        tomography.
        """
        base_circ = Circuit(2)
        original_unitary = base_circ.U_full
        StateTomography(1, base_circ, experiment)
        assert pytest.approx(original_unitary) == base_circ.U
