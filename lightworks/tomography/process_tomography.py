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

from typing import Callable

import numpy as np

from .. import qubit
from ..sdk.circuit import Circuit
from ..sdk.state import State
from .state_tomography import StateTomography as StateTomo

TOMO_INPUTS = ["0", "1", "+", "R"]

r_transform = qubit.H()
r_transform.add(qubit.S())

INPUT_MAPPING: dict[str, tuple[State, Circuit]] = {
    "0": (State([1, 0]), qubit.I()),
    "1": (State([0, 1]), qubit.I()),
    "+": (State([1, 0]), qubit.H()),
    "R": (State([1, 0]), r_transform),
}


class ProcessTomography:
    """
    Desc
    """

    def __init__(
        self,
        n_qubits: int,
        base_circuit: Circuit,
        experiment: Callable,
        experiment_args: list | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.base_circuit = base_circuit
        self.experiment = experiment
        self.experiment_args = experiment_args

    def process(self) -> np.ndarray:
        """
        Desc
        """
        all_inputs = list(TOMO_INPUTS)
        for _ in range(self.n_qubits - 1):
            all_inputs = [i1 + i2 for i1 in all_inputs for i2 in TOMO_INPUTS]

        req_measurements, result_mapping = StateTomo._get_required_measurements(
            self.n_qubits
        )

        all_circuits = []
        all_input_states = []

        for in_state in all_inputs:
            for meas in req_measurements:
                circ, state = self._create_circuit_and_input(in_state, meas)
                all_circuits.append(circ)
                all_input_states.append(state)

        experiment_args = (
            self.experiment_args if self.experiment_args is not None else []
        )

        results = self.experiment(
            all_circuits, all_input_states, *experiment_args
        )

        # Sorted results into each input/measurement combination
        n_per_in = len(req_measurements)
        sorted_results = {
            in_state: dict(
                zip(
                    req_measurements, results[n_per_in * i : n_per_in * (i + 1)]
                )
            )
            for i, in_state in enumerate(all_inputs)
        }
        # Expand results to include all of the required measurements
        full_results = {}
        for in_state, res in sorted_results.items():
            full_res = {
                meas: res[result_mapping[meas]]
                for meas in StateTomo._get_all_measurements(self.n_qubits)
            }
            full_results[in_state] = full_res

        return {
            in_state: StateTomo._calculate_density_matrix(self.n_qubits, res)
            for in_state, res in full_results.items()
        }

    def _create_circuit_and_input(
        self, input_op: str, output_op: str
    ) -> tuple[Circuit, State]:
        """
        Desc
        """
        in_state = State([])
        circ = Circuit(self.base_circuit.input_modes)

        for i, op in enumerate(input_op):
            in_state += INPUT_MAPPING[op][0]
            circ.add(INPUT_MAPPING[op][1], 2 * i)

        circ.add(self.base_circuit)

        for i, op in enumerate(output_op):
            circ.add(StateTomo._get_measurement_operator(op), 2 * i)

        return circ, in_state
