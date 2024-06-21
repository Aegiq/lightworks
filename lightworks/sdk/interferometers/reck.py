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

from ..circuit import Circuit
from .decomposition import reck_decomposition


class Reck:
    """
    Desc
    """

    def __init__(self) -> None:
        # TODO: Will implement some sort of error model here.

        return

    def map(self, circuit: Circuit) -> None:
        """
        Maps a provided circuit onto the interferometer.
        """
        phase_map, end_phases = reck_decomposition(circuit.U)

        n_modes = circuit.n_modes

        mapped_circuit = Circuit(n_modes)
        # Build circuit here
        for i in range(n_modes - 1):
            for j in range(0, n_modes - 1 - i, 1):
                coord = f"{j + 2 * i}_{j}"
                mapped_circuit.add_ps(j, phase_map["ps_" + coord])
                mapped_circuit.add_bs(j)
                mapped_circuit.add_ps(j + 1, phase_map["bs_" + coord])
                mapped_circuit.add_bs(j)
        for i in range(n_modes):
            mapped_circuit.add_ps(i, end_phases[i])

        # TODO: Copy over heralds
        return mapped_circuit
