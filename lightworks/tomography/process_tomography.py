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
from numpy.typing import NDArray

from lightworks.sdk.circuit import PhotonicCircuit
from lightworks.sdk.state import State

from .experiments import ProcessTomographyExperiment, ProcessTomographyList
from .mappings import INPUT_MAPPING, MEASUREMENT_MAPPING, RHO_MAPPING
from .tomography import _Tomography
from .utils import (
    TomographyDataError,
    _calculate_density_matrix,
    _check_target_process,
    _combine_all,
    _get_required_tomo_measurements,
    _get_tomo_measurements,
    state_fidelity,
)


class _ProcessTomography(_Tomography):
    """
    Process tomography base class, implements some of the common methods
    required across different approaches.
    """

    _tomo_inputs: tuple[str, ...] = ("Z+", "Z-", "X+", "Y+")

    def get_experiments(self) -> ProcessTomographyList:
        """
        Generates all required tomography experiments for performing a process
        tomography algorithm.
        """
        inputs = self._full_input_basis()
        req_measurements, _ = _get_required_tomo_measurements(self.n_qubits)
        # Determine required input states and circuits
        experiments = ProcessTomographyList()
        for in_basis in inputs:
            for meas_basis in req_measurements:
                circ, state = self._create_circuit_and_input(
                    in_basis, meas_basis
                )
                experiments.append(
                    ProcessTomographyExperiment(
                        circuit=circ,
                        input_state=state,
                        input_basis=in_basis,
                        measurement_basis=meas_basis,
                    )
                )

        return experiments

    def process_state_fidelities(
        self,
        data: list[dict[State, int]] | dict[tuple[str, str], dict[State, int]],
        target_process: NDArray[np.complex128],
    ) -> dict[str, float]:
        """
        Calculates the density matrix of the state created for each input to the
        system and finds the fidelity relative to the expected matrix.

        Args:

            data (list | dict) : The collected measurement data. If a list then
                this should match the order the experiments were provided, and
                if a dictionary, then each key should be tuple of the input and
                measurement basis.

            target_process (np.ndarray) : The unitary matrix corresponding to
                the target process. The dimension of this should be 2^n_qubits.

        Returns:

            float : The calculated fidelity.

        """
        target_process = _check_target_process(target_process, self.n_qubits)
        rhos = self.process_density_matrices(data)
        rho_exp = self.get_expected_density_matrices(target_process)
        return {i: state_fidelity(r, rho_exp[i]) for i, r in rhos.items()}

    def process_density_matrices(
        self,
        data: list[dict[State, int]] | dict[tuple[str, str], dict[State, int]],
    ) -> dict[str, NDArray[np.complex128]]:
        """
        Calculate and returns the density matrix of the state created for each
        input to the system.

        Args:

            data (list | dict) : The collected measurement data. If a list then
                this should match the order the experiments were provided, and
                if a dictionary, then each key should be tuple of the input and
                measurement basis.

        Returns:

            dict : A dictionary of the calculated density matrix for each input
                to the system.

        """
        # Run all required tomography experiments
        results = self._convert_tomography_data(data)
        # Sorted results per input
        remapped_results: dict[str, dict[str, dict[State, int]]] = {
            k[0]: {} for k in results
        }
        for (k1, k2), r in results.items():
            remapped_results[k1][k2] = r
        # Calculate density matrices
        return {
            i: _calculate_density_matrix(remapped_results[i], self.n_qubits)
            for i in self._full_input_basis()
        }

    def get_expected_density_matrices(
        self, target_process: NDArray[np.complex128]
    ) -> dict[str, NDArray[np.complex128]]:
        """
        Calculates the expected output density matrices for each input when a
        target process is applied.

        Args:

            target_process (np.ndarray) : The unitary matrix corresponding to
                the target process. The dimension of this should be 2^n_qubits.

        Returns:

            dict : The calculated density matrices for each input.

        """
        target_process = _check_target_process(target_process, self.n_qubits)
        input_rhos = _combine_all(
            {k: v for k, v in RHO_MAPPING.items() if k in self._tomo_inputs},
            self.n_qubits,
        )
        return {
            i: target_process @ input_rhos[i] @ np.conj(target_process.T)
            for i in self._full_input_basis()
        }

    def _full_input_basis(self) -> list[str]:
        """
        Calculates and returns the full input basis used in a tomography
        experiment.
        """
        return _combine_all(list(self._tomo_inputs), self.n_qubits)

    def _create_circuit_and_input(
        self, input_op: str, output_op: str
    ) -> tuple[PhotonicCircuit, State]:
        """
        Creates the required circuit and input state to achieve a provided input
        and output operation.
        """
        in_state = State([])
        circ = PhotonicCircuit(self.base_circuit.input_modes)
        # Input operation
        for i, op in enumerate(input_op.split(",")):
            in_state += INPUT_MAPPING[op][0]
            circ.add(INPUT_MAPPING[op][1], 2 * i)
        # Add base circuit
        circ.add(self.base_circuit)
        # Measurement operation
        for i, op in enumerate(output_op.split(",")):
            circ.add(MEASUREMENT_MAPPING[op], 2 * i)
        return circ, in_state

    def _convert_tomography_data(
        self,
        results: list[dict[State, int]]
        | dict[tuple[str, str], dict[State, int]],
    ) -> dict[tuple[str, str], dict[State, int]]:
        """
        Processes the provided tomography data, validating all expected
        measurements are present and converting it into a full dataset including
        previously removed redundant measurements (in the 'I' basis).
        """
        # Re-generate all tomography data
        inputs = _combine_all(list(self._tomo_inputs), self.n_qubits)
        req_measurements, result_mapping = _get_required_tomo_measurements(
            self.n_qubits
        )
        if not isinstance(results, dict):
            if len(results) != len(inputs) * len(req_measurements):
                msg = (
                    f"Number of results ({len(results)}) did not match the "
                    f"expected number ({len(inputs) * len(req_measurements)}) "
                    "for the target tomography algorithm."
                )
                raise TomographyDataError(msg)
            # Sort results into each input/measurement combination
            num_per_in = len(req_measurements)
            sorted_results = {
                in_state: dict(
                    zip(
                        req_measurements,
                        results[num_per_in * i : num_per_in * (i + 1)],
                        strict=True,
                    )
                )
                for i, in_state in enumerate(inputs)
            }
        else:
            sorted_results = {}
            missing = []
            for in_state in inputs:
                sorted_results[in_state] = {}
                for meas in req_measurements:
                    if (in_state, meas) in results:
                        sorted_results[in_state][meas] = results[in_state, meas]
                    else:
                        missing.append((in_state, meas))
            if missing:
                msg = (
                    "One or more expected keys were detected to be missing "
                    f"from the results dictionary. Missing keys were {missing}."
                )
                raise TomographyDataError(msg)
        # Expand results to include all of the required measurements
        full_results = {}
        for in_state, res in sorted_results.items():
            for meas in _get_tomo_measurements(self.n_qubits):
                full_results[in_state, meas] = res[result_mapping[meas]]
        return full_results
