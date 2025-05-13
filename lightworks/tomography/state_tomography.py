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

from .experiments import StateTomographyExperiment, StateTomographyList
from .mappings import MEASUREMENT_MAPPING
from .tomography import _Tomography
from .utils import (
    TomographyDataError,
    _calculate_density_matrix,
    _get_required_tomo_measurements,
    _get_tomo_measurements,
    state_fidelity,
)


class StateTomography(_Tomography):
    """
    Generates the required circuit and performs data processing for the
    calculation of the density matrix of a state.

    Args:

        n_qubits (int) : The number of qubits that will be used as part of the
            tomography.

        base_circuit (PhotonicCircuit) : An initial circuit which produces the
            required output state and can be modified for performing tomography.
            It is required that the number of circuit input modes equals 2 * the
            number of qubits.

    """

    @property
    def rho(self) -> NDArray[np.complex128]:
        """
        The most recently calculated density matrix.
        """
        if not hasattr(self, "_rho"):
            raise AttributeError(
                "Density matrix has not yet been calculated, this can be "
                "achieved with the process method."
            )
        return self._rho

    def get_experiments(self) -> StateTomographyList:
        """
        Generates all required tomography experiments for performing a process
        tomography algorithm.
        """
        req_measurements, _ = _get_required_tomo_measurements(self.n_qubits)

        # Generate all circuits and run experiment
        experiments = StateTomographyList()
        for gates in req_measurements:
            experiments.append(
                StateTomographyExperiment(
                    circuit=self._create_circuit(
                        [MEASUREMENT_MAPPING[g] for g in gates.split(",")]
                    ),
                    measurement_basis=gates,
                )
            )

        return experiments

    def process(
        self,
        data: list[dict[State, int]] | dict[str, dict[State, int]],
    ) -> NDArray[np.complex128]:
        """
        Performs the state tomography process with the configured elements to
        calculate the density matrix of the output state.

        Returns:

            np.ndarray : The calculated density matrix from the state tomography
                process.

        """
        req_measurements, result_mapping = _get_required_tomo_measurements(
            self.n_qubits
        )
        # Convert results into dictionary and then mapping to full set of
        # measurements
        if not isinstance(data, dict):
            if len(data) != len(req_measurements):
                msg = (
                    f"Number of results ({len(data)}) did not match the "
                    f"expected number ({len(req_measurements)}) for the target "
                    "tomography algorithm."
                )
                raise TomographyDataError(msg)
            results_dict = dict(zip(req_measurements, data, strict=True))
        else:
            missing = [meas for meas in req_measurements if meas not in data]
            if missing:
                msg = (
                    "One or more expected keys were detected to be missing "
                    f"from the results dictionary. Missing keys were {missing}."
                )
                raise TomographyDataError(msg)
            results_dict = dict(data)
        results_dict = {
            c: results_dict[result_mapping[c]]
            for c in _get_tomo_measurements(self.n_qubits)
        }

        self._rho = _calculate_density_matrix(results_dict, self.n_qubits)
        return self.rho

    def fidelity(self, rho_exp: NDArray[np.complex128]) -> float:
        """
        Calculates the fidelity of the calculated quantum state against the
        expected density matrix for the state.

        Args:

            rho_exp (np.ndarray) : The expected density matrix.

        Returns:

            float : The calculated fidelity value.

        """
        return state_fidelity(self.rho, rho_exp)

    def _create_circuit(
        self, measurement_operators: list[PhotonicCircuit]
    ) -> PhotonicCircuit:
        """
        Creates a copy of the assigned base circuit and applies the list of
        measurement circuits to each pair of dual-rail encoded qubits.

        Args:

            measurement_operators (list) : A list of 2 mode circuits which act
                as measurement operators to apply to the system.

        Returns:

            PhotonicCircuit : A modified copy of the base circuit with required
                operations.

        """
        circuit = self.base_circuit.copy()
        # Check number of circuits is correct
        if len(measurement_operators) != self.n_qubits:
            msg = (
                "Number of operators should match number of qubits "
                f"({self.n_qubits})."
            )
            raise ValueError(msg)
        # Add each and then return
        for i, op in enumerate(measurement_operators):
            circuit.add(op, 2 * i)

        return circuit
