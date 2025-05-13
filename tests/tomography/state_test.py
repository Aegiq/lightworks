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

from lightworks import (
    PhotonicCircuit,
    PostSelection,
    Sampler,
    State,
    Unitary,
    qubit,
    random_unitary,
)
from lightworks.emulator import Backend
from lightworks.tomography import StateTomography
from lightworks.tomography.experiments import (
    StateTomographyExperiment,
    StateTomographyList,
)
from lightworks.tomography.state_tomography import MEASUREMENT_MAPPING


def run_experiments(experiments):
    """
    Experiment function with ability to specify the input state used.
    """
    # Find number of qubits using available input modes.
    n_qubits = int(experiments[0].circuit.input_modes / 2)
    n_samples = 25000
    post_selection = PostSelection()
    for i in range(n_qubits):
        post_selection.add((2 * i, 2 * i + 1), 1)
    results = []
    backend = Backend("slos")
    for exp in experiments:
        sampler = Sampler(
            exp.circuit,
            State([1, 0] * n_qubits),
            n_samples,
            post_selection=post_selection,
            random_seed=29,
        )
        results.append(backend.run(sampler))
    return results


class TestStateTomography:
    """
    Unit tests for state tomography class.
    """

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_basic_state(self, n_qubits):
        """
        Checks correct density matrix is produced when performing tomography on
        the |0> X n_qubits state.
        """
        base_circ = PhotonicCircuit(n_qubits * 2)
        tomo = StateTomography(n_qubits, base_circ)
        experiments = tomo.get_experiments()
        data = run_experiments(experiments)
        rho = tomo.process(data)
        rho_exp = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        rho_exp[0, 0] = 1
        assert rho == pytest.approx(rho_exp, abs=1e-2)
        assert tomo.fidelity(rho_exp) == pytest.approx(1, 1e-3)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_ghz_state(self, n_qubits):
        """
        Checks correct density matrix is produced when performing tomography on
        the n_qubit GHZ state.
        """
        base_circ = PhotonicCircuit(n_qubits * 2)
        base_circ.add(qubit.H())
        for i in range(n_qubits - 1):
            base_circ.add(qubit.CNOT(), 2 * i)
        tomo = StateTomography(n_qubits, base_circ)
        experiments = tomo.get_experiments()
        data = run_experiments(experiments)
        rho = tomo.process(data)
        rho_exp = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        rho_exp[0, 0] = 0.5
        rho_exp[0, -1] = 0.5
        rho_exp[-1, 0] = 0.5
        rho_exp[-1, -1] = 0.5
        assert rho == pytest.approx(rho_exp, abs=1e-2)
        assert tomo.fidelity(rho_exp) == pytest.approx(1, 1e-3)

    @pytest.mark.parametrize("to_shuffle", [True, False])
    def test_ghz_state_dict(self, to_shuffle):
        """
        Checks correct density matrix is produced when performing tomography on
        the n_qubit GHZ state and returning results as a dictionary.
        """
        n_qubits = 2
        base_circ = PhotonicCircuit(n_qubits * 2)
        base_circ.add(qubit.H())
        for i in range(n_qubits - 1):
            base_circ.add(qubit.CNOT(), 2 * i)
        tomo = StateTomography(n_qubits, base_circ)
        experiments = tomo.get_experiments()
        data = run_experiments(experiments)
        # Convert data to dict
        data_dict = dict(
            zip(experiments.all_measurement_basis, data, strict=True)
        )
        if to_shuffle:
            keys = list(data_dict.keys())
            shuffle(keys)
            data_dict = {k: data_dict[k] for k in keys}
        rho = tomo.process(data_dict)
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
            StateTomography(2, PhotonicCircuit(n_modes))

    @pytest.mark.parametrize("value", [1.5, "2", None, True])
    def test_n_qubits_must_be_integer(self, value):
        """
        Checks value of n_qubits must be an integer.
        """
        with pytest.raises(TypeError, match="qubits"):
            StateTomography(value, PhotonicCircuit(4))

    @pytest.mark.parametrize(
        "value", [PhotonicCircuit(4).U, [1, 2, 3], None, True]
    )
    def test_base_circuit_must_be_circuit(self, value):
        """
        Checks value of base_circuit must be a PhotonicCircuit object.
        """
        with pytest.raises(TypeError, match="circuit"):
            StateTomography(2, value)

    def test_density_mat_before_calc(self):
        """
        Checks an error is raised if the rho attribute is called before
        tomography is performed.
        """
        tomo = StateTomography(2, PhotonicCircuit(4))
        with pytest.raises(AttributeError):
            tomo.rho  # noqa: B018

    def test_fidleity_before_calc(self):
        """
        Checks an error is raised if a user attempts to calculate fidelity
        before performing tomography.
        """
        tomo = StateTomography(2, PhotonicCircuit(4))
        with pytest.raises(AttributeError):
            tomo.fidelity(np.identity(2))

    def test_base_circuit_unmodified(self):
        """
        Confirms base circuit is unmodified when performing single qubit
        tomography.
        """
        base_circ = PhotonicCircuit(2)
        original_unitary = base_circ.U_full
        tomo = StateTomography(1, base_circ)
        experiments = tomo.get_experiments()
        data = run_experiments(experiments)
        tomo.process(data)
        assert pytest.approx(original_unitary) == base_circ.U

    def test_density_matrix_matches(self):
        """
        Confirms density matrix property returns correct value.
        """
        base_circ = PhotonicCircuit(2)
        tomo = StateTomography(1, base_circ)
        experiments = tomo.get_experiments()
        data = run_experiments(experiments)
        rho1 = tomo.process(data)
        rho2 = tomo.rho
        assert (rho1 == rho2).all()

    @pytest.mark.parametrize("operator", ["I", "X", "Y", "Z"])
    def test_circuit_produces_correct_circuit(self, operator):
        """
        Confirms that create circuit function correctly modifies a base circuit.
        """
        base_circ = Unitary(random_unitary(4))
        op = MEASUREMENT_MAPPING[operator]
        # Get tomography circuit
        tomo = StateTomography(2, base_circ)
        tomo_circ = tomo._create_circuit([MEASUREMENT_MAPPING["I"], op])
        # Modify base circuit and compare
        base_circ.add(MEASUREMENT_MAPPING[operator], 2)
        # Compare
        assert tomo_circ.U_full == pytest.approx(base_circ.U_full)

    def test_circuit_enforces_measurement_length(self):
        """
        Checks that create circuit function will raise an error if the
        measurement string is the wrong length.
        """
        tomo = StateTomography(2, PhotonicCircuit(4))
        with pytest.raises(ValueError):
            tomo._create_circuit("XYZ")

    def test_experiments_length(self):
        """
        Checks number of elements returned by StateTomography is the correct
        length.
        """
        tomo = StateTomography(2, qubit.CNOT())
        experiments = tomo.get_experiments()
        assert len(experiments) == 9

    def test_experiments_list(self):
        """
        Checks that data returned by StateTomography get_experiments is a
        StateTomographyList.
        """
        tomo = StateTomography(2, qubit.CNOT())
        experiments = tomo.get_experiments()
        assert isinstance(experiments, StateTomographyList)

    def test_experiments_in_list_are_experiments(self):
        """
        Checks that data returned by StateTomography get_experiments in a list
        of experiments.
        """
        tomo = StateTomography(2, qubit.CNOT())
        experiments = tomo.get_experiments()
        for exp in experiments:
            assert isinstance(exp, StateTomographyExperiment)

    @pytest.mark.parametrize(
        ("n1", "n2"),
        [
            ("all_circuits", "circuit"),
            ("all_measurement_basis", "measurement_basis"),
        ],
    )
    def test_all_methods(self, n1, n2):
        """
        Checks that the all methods from StateTomographyList correctly returns
        the expected list of values in the correct order.
        """
        tomo = StateTomography(2, qubit.CNOT())
        experiments = tomo.get_experiments()
        for exp, quantity in zip(
            experiments, getattr(experiments, n1), strict=True
        ):
            assert getattr(exp, n2) == quantity
