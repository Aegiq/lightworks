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

from lightworks.sdk.circuit import PhotonicCircuit

from .decomposition import reck_decomposition
from .error_model import ErrorModel


class Reck:
    """
    Maps a provided PhotonicCircuit to a Reck interferometer, implementing the
    configured error model across all components.

    Args:

        error_model (ErrorModel | None) : An error model object which implements
            variation of the interferometer parameters. If not specified then
            an ideal system will be assumed.

    """

    def __init__(self, error_model: ErrorModel | None = None) -> None:
        # If no error model provided then set to default
        if error_model is None:
            error_model = ErrorModel()
        self.error_model = error_model

    @property
    def error_model(self) -> ErrorModel:
        """Returns currently used error model for the system."""
        return self.__error_model

    @error_model.setter
    def error_model(self, value: ErrorModel) -> None:
        if not isinstance(value, ErrorModel):
            raise TypeError("error_model should be an ErrorModel object.")
        self.__error_model = value

    def map(
        self, circuit: PhotonicCircuit, seed: int | None = None
    ) -> PhotonicCircuit:
        """
        Maps a provided circuit onto the interferometer.

        Args:

            circuit (PhotonicCircuit) : The circuit to map onto the reck
                interferometer.

            seed (int | None) : Set a random seed to produce identical results
                when implementing a probabilistic error model.

        Returns:

            PhotonicCircuit : The created Reck interferometer which implements
                the programmed transformation.

        """
        # Reset error model seed
        self.error_model._set_random_seed(seed)
        # Invert unitary so reck layout starts with fewest elements on mode 0
        unitary = np.flip(circuit.U, axis=(0, 1))
        phase_map, end_phases = reck_decomposition(unitary)
        phase_map = {
            k: (v + self.error_model.get_phase_offset()) % (2 * np.pi)
            for k, v in phase_map.items()
        }
        end_phases = [
            (p + self.error_model.get_phase_offset()) % (2 * np.pi)
            for p in end_phases
        ]

        # Build circuit with required mode number
        n_modes = circuit.n_modes
        mapped_circuit = PhotonicCircuit(n_modes)
        for i in range(n_modes - 1):
            for j in range(0, n_modes - 1 - i, 1):
                # Find coordinates + mode from i & j values
                coord = f"{j + 2 * i}_{j}"
                mode = n_modes - j - 2
                mapped_circuit.barrier([mode, mode + 1])
                mapped_circuit.ps(mode + 1, phase_map["ps_" + coord])
                mapped_circuit.bs(
                    mode, reflectivity=self.error_model.get_bs_reflectivity()
                )
                mapped_circuit.ps(mode, phase_map["bs_" + coord])
                mapped_circuit.bs(
                    mode,
                    reflectivity=self.error_model.get_bs_reflectivity(),
                    loss=self.error_model.get_loss(),
                )
        mapped_circuit.barrier()
        # Apply residual phases at the end
        for i in range(n_modes):
            mapped_circuit.ps(n_modes - i - 1, end_phases[i])

        # Add any heralds from the original circuit
        heralds = circuit.heralds
        for m1, m2 in zip(heralds.input, heralds.output, strict=True):
            mapped_circuit.herald(
                (m1, m2), (heralds.input[m1], heralds.output[m2])
            )

        return mapped_circuit
