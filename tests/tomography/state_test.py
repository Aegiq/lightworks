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
from lightworks.tomography.state_tomography import MEASUREMENT_MAPPING


def experiment_args(circuits, input_state):
    """
    Experiment function with ability to specify the input state used.
    """
    # Find number of qubits using available input modes.
    n_qubits = int(circuits[0].input_modes / 2)
    n_samples = 25000
    post_selection = PostSelection()
    for i in range(n_qubits):
        post_selection.add((2 * i, 2 * i + 1), 1)
    results = []
    backend = Backend("slos")
    for circ in circuits:
        sampler = Sampler(
            circ,
            input_state,
            n_samples,
            post_selection=post_selection,
            random_seed=29,
        )
        results.append(backend.run(sampler))
    return results


def experiment(circuits):
    """
    Experiment function for testing state tomography functionality is correct.
    """
    n_qubits = int(circuits[0].input_modes / 2)
    return experiment_args(circuits, State([1, 0] * n_qubits))


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
        tomo = StateTomography(n_qubits, base_circ, experiment)
        rho = tomo.process()
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
            StateTomography(2, PhotonicCircuit(n_modes), experiment)

    @pytest.mark.parametrize("value", [1.5, "2", None, True])
    def test_n_qubits_must_be_integer(self, value):
        """
        Checks value of n_qubits must be an integer.
        """
        with pytest.raises(TypeError, match="qubits"):
            StateTomography(value, PhotonicCircuit(4), experiment)

    @pytest.mark.parametrize(
        "value", [PhotonicCircuit(4).U, [1, 2, 3], None, True]
    )
    def test_base_circuit_must_be_circuit(self, value):
        """
        Checks value of base_circuit must be a PhotonicCircuit object.
        """
        with pytest.raises(TypeError, match="circuit"):
            StateTomography(2, value, experiment)

    @pytest.mark.parametrize("value", [PhotonicCircuit(4), 4, None])
    def test_experiment_must_be_function(self, value):
        """
        Checks value of experiment must be a function.
        """
        with pytest.raises(TypeError, match="experiment"):
            StateTomography(2, PhotonicCircuit(4), value)

    def test_density_mat_before_calc(self):
        """
        Checks an error is raised if the rho attribute is called before
        tomography is performed.
        """
        tomo = StateTomography(2, PhotonicCircuit(4), experiment)
        with pytest.raises(AttributeError):
            tomo.rho  # noqa: B018

    def test_fidleity_before_calc(self):
        """
        Checks an error is raised if a user attempts to calculate fidelity
        before performing tomography.
        """
        tomo = StateTomography(2, PhotonicCircuit(4), experiment)
        with pytest.raises(AttributeError):
            tomo.fidelity(np.identity(2))

    def test_base_circuit_unmodified(self):
        """
        Confirms base circuit is unmodified when performing single qubit
        tomography.
        """
        base_circ = PhotonicCircuit(2)
        original_unitary = base_circ.U_full
        StateTomography(1, base_circ, experiment).process()
        assert pytest.approx(original_unitary) == base_circ.U

    def test_density_matrix_matches(self):
        """
        Confirms density matrix property returns correct value.
        """
        base_circ = PhotonicCircuit(2)
        tomo = StateTomography(1, base_circ, experiment)
        rho1 = tomo.process()
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
        tomo = StateTomography(2, base_circ, experiment)
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
        tomo = StateTomography(2, PhotonicCircuit(4), experiment)
        with pytest.raises(ValueError):
            tomo._create_circuit("XYZ")

    def test_experiment_args(self):
        """
        Checks that experiment arguments can be provided to StateTomography.
        """
        tomo = StateTomography(
            1,
            PhotonicCircuit(2),
            experiment_args,
            experiment_args=[State([1, 0])],
        )
        rho = tomo.process()
        rho_exp = np.zeros((2, 2), dtype=complex)
        rho_exp[0, 0] = 1
        assert rho == pytest.approx(rho_exp, abs=1e-2)
        assert tomo.fidelity(rho_exp) == pytest.approx(1, 1e-3)
