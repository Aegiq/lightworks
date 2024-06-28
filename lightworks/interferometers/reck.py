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

from ..sdk.circuit import Circuit
from .decomposition import reck_decomposition
from .error_model import ErrorModel


class Reck:
    """
    Used for mapping a provided Circuit to a Reck interferometer.

    Args:

        error_model (ErrorModel) : An error model object which implements
            variation of the interferometer parameters.

    """

    def __init__(self, error_model: ErrorModel | None = None) -> None:
        # If no error model provided then set to default
        if error_model is None:
            error_model = ErrorModel()
        self.error_model = error_model

        return

    @property
    def error_model(self) -> ErrorModel:
        """Returns currently used error model for the system."""
        return self.__error_model

    @error_model.setter
    def error_model(self, value: ErrorModel) -> None:
        if not isinstance(value, ErrorModel):
            raise TypeError("error_model should be an ErrorModel object.")
        self.__error_model = value

    def map(self, circuit: Circuit, seed: int | None = None) -> Circuit:
        """
        Maps a provided circuit onto the interferometer.

        Args:

            circuit (Circuit) : The circuit to map onto the reck interferometer.

            seed (int | None) : Set a random seed to produce identical results
                when implementing a probabilistic error model.

        Returns:

            Circuit : The created Reck interferometer which implements the
                programmed transformation.

        """
        # Reset error model seed
        self.error_model._set_random_seed(seed)
        # Invert unitary so reck layout starts with fewest elements on mode 0
        unitary = np.flip(circuit.U, axis=(0, 1))
        phase_map, end_phases = reck_decomposition(unitary)
        phase_map = {k: v % (2 * np.pi) for k, v in phase_map.items()}
        end_phases = [p % (2 * np.pi) for p in end_phases]

        # Build circuit with required mode number
        n_modes = circuit.n_modes
        mapped_circuit = Circuit(n_modes)
        for i in range(n_modes - 1):
            for j in range(0, n_modes - 1 - i, 1):
                # Find coordinates + mode from i & j values
                coord = f"{j + 2 * i}_{j}"
                mode = n_modes - j - 2
                mapped_circuit.add_barrier([mode, mode + 1])
                mapped_circuit.add_ps(mode + 1, phase_map["ps_" + coord])
                mapped_circuit.add_bs(
                    mode, reflectivity=self.error_model.get_bs_reflectivity()
                )
                mapped_circuit.add_ps(mode, phase_map["bs_" + coord])
                mapped_circuit.add_bs(
                    mode,
                    reflectivity=self.error_model.get_bs_reflectivity(),
                    loss=self.error_model.get_loss(),
                )
        mapped_circuit.add_barrier()
        # Apply residual phases at the end
        for i in range(n_modes):
            mapped_circuit.add_ps(n_modes - i - 1, end_phases[i])

        # Add any heralds from the original circuit
        heralds = circuit.heralds
        for m1, m2 in zip(heralds["input"], heralds["output"]):
            # NOTE: Could be errors if input and output photon numbers don't
            # match - this shouldn't happen but for now check just in case.
            if heralds["input"][m1] != heralds["output"][m2]:
                raise RuntimeError("Mismatching heralding numbers detected.")
            mapped_circuit.add_herald(heralds["input"][m1], m1, m2)

        return mapped_circuit
