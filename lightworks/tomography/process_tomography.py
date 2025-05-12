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


from lightworks.sdk.circuit import PhotonicCircuit
from lightworks.sdk.state import State

from .mappings import INPUT_MAPPING, MEASUREMENT_MAPPING
from .utils import (
    TomographyExperiment,
    TomographyList,
    _combine_all,
    _get_required_tomo_measurements,
    _get_tomo_measurements,
)


class ProcessTomography:
    """
    Process tomography base class, implements some of the common methods
    required across different approaches.

    Args:

        n_qubits (int) : The number of qubits that will be used as part of the
            tomography.

        base_circuit (PhotonicCircuit) : An initial circuit which produces the
            required output state and can be modified for performing tomography.
            It is required that the number of circuit input modes equals 2 * the
            number of qubits.

        experiment (Callable) : A function for performing the required
            tomography experiments. This should accept a list of circuits and a
            list of inputs and then return a list of results to process.

        experiment_args (list | None) : Optionally provide additional arguments
            which will be passed directly to the experiment function.

    """

    _tomo_inputs: tuple[str, ...] = ("Z+", "Z-", "X+", "Y+")

    def __init__(
        self,
        n_qubits: int,
        base_circuit: PhotonicCircuit,
    ) -> None:
        # Type check inputs
        if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
            raise TypeError("Number of qubits should be an integer.")
        if not isinstance(base_circuit, PhotonicCircuit):
            raise TypeError("Base circuit should be a circuit object.")

        if 2 * n_qubits != base_circuit.input_modes:
            msg = (
                "Number of circuit input modes does not match the amount "
                "required for the specified number of qubits, expected "
                f"{2 * n_qubits}."
            )
            raise ValueError(msg)

        self._n_qubits = n_qubits
        self._base_circuit = base_circuit

    @property
    def base_circuit(self) -> PhotonicCircuit:
        """
        The base circuit which is to be modified as part of the tomography
        calculations.
        """
        return self._base_circuit

    @property
    def n_qubits(self) -> int:
        """
        The number of qubits within the system.
        """
        return self._n_qubits

    def get_experiments(self) -> TomographyList:
        """
        Generates all required tomography experiments for performing a process
        tomography algorithm.
        """
        inputs = self._full_input_basis()
        req_measurements, _ = _get_required_tomo_measurements(self.n_qubits)
        # Determine required input states and circuits
        experiments = TomographyList()
        for in_basis in inputs:
            for meas_basis in req_measurements:
                circ, state = self._create_circuit_and_input(
                    in_basis, meas_basis
                )
                experiments.append(
                    TomographyExperiment(circ, state, in_basis, meas_basis)
                )

        return experiments

    def _full_input_basis(self) -> list[str]:
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
        # Re-generate all tomography data
        inputs = _combine_all(list(self._tomo_inputs), self.n_qubits)
        req_measurements, result_mapping = _get_required_tomo_measurements(
            self.n_qubits
        )
        if not isinstance(results, dict):
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
            sorted_results = dict(results)  # type: ignore [arg-type]
        # Expand results to include all of the required measurements
        full_results = {}
        for in_state, res in sorted_results.items():
            for meas in _get_tomo_measurements(self.n_qubits):
                full_results[in_state, meas] = res[result_mapping[meas]]
        return full_results
