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

"""
Contains a variety of qubit two components, designed for implementing required
qubit processing functionality across pairs of dual-rail encoded qubits in
lightworks.
"""

import numpy as np

from ...sdk.circuit import PhotonicCircuit, Unitary
from ...sdk.utils import permutation_mat_from_swaps_dict
from .single_qubit_gates import H


class CZ(PhotonicCircuit):
    """
    Post-selected CZ gate that acts across two dual-rail encoded qubits. This
    gate occupies a total of 6 modes, where modes 0 & 5 are used for 0 photon
    heralds, modes 1 & 2 correspond to the 0 & 1 states of the first qubit and
    modes 3 & 4 correspond to the 0 & 1 states of the second qubit. This
    gate requires additional post-selection in which only one photon should be
    measured across each of the pairs of modes which encode a qubit.
    """

    def __init__(self) -> None:
        u_bs = np.array([[-1, 2**0.5], [2**0.5, 1]]) / 3**0.5
        u_a = np.identity(6, dtype=complex)
        for i in range(0, 6, 2):
            u_a[i : i + 2, i : i + 2] = u_bs[:, :]
        u_a[3, :] = -u_a[3, :]
        unitary = Unitary(u_a, label="CZ")
        unitary.herald(0, 0, 0)
        unitary.herald(0, 5, 5)

        super().__init__(4)
        self.add(unitary, 0, group=True, name="CZ")


class CNOT(PhotonicCircuit):
    """
    Post-selected CNOT gate that acts across two dual-rail encoded qubits. This
    gate occupies a total of 6 modes, where modes 0 & 5 are used for 0 photon
    heralds, modes 1 & 2 correspond to the 0 & 1 states of the first qubit and
    modes 3 & 4 correspond to the 0 & 1 states of the second qubit. This gate
    requires additional post-selection in which only one photon should be
    measured across each of the pairs of modes which encode a qubit.

    Args:

        target_qubit (int, optional) : Sets which of the two qubits is used as
            the target qubit for the gate. Should be either 0 or 1.

    """

    def __init__(self, target_qubit: int = 1) -> None:
        if target_qubit not in [0, 1]:
            raise ValueError(
                "target_qubit setting must have a value of either 0 or 1."
            )

        super().__init__(4)

        # Create CNOT from combination of H and CZ
        circ = PhotonicCircuit(4)
        circ.add(H(), 2 * target_qubit)
        circ.add(CZ(), 0)
        circ.add(H(), 2 * target_qubit)

        name = f"CNOT ({1 - target_qubit}, {target_qubit})"
        self.add(circ, 0, group=True, name=name)


class CZ_Heralded(PhotonicCircuit):  # noqa: N801
    """
    Heralded version of the CZ gate which acts across two dual-rail encoded
    qubits, using two NS gates with ancillary photons to herald the success of
    the transformation. This gate occupies 8 modes, where modes 0 & 7 are used
    as 0 photon heralds, modes 1 & 6 are used as 1 photon heralds, mode 2 & 3
    correspond to the 0 & 1 states of the first qubit and modes 4 & 5 correspond
    to the 0 & 1 states of the second qubit. The heralded gate does not require
    any post-selection on the output qubits, other than that they are not lost
    (i.e a total of 4 photons should be measured at the output of the system),
    allowing it to be cascaded with other two qubit gates.
    """

    def __init__(self) -> None:
        u_a = np.identity(8, dtype=complex)

        u_ns = np.array(
            [
                [1 - 2**0.5, 2**-0.25, (3 / (2**0.5) - 2) ** 0.5],
                [2**-0.25, 0.5, 0.5 - 2**-0.5],
                [(3 / (2**0.5) - 2) ** 0.5, 0.5 - 2**-0.5, 2**0.5 - 0.5],
            ]
        )
        u_a[1:4, 1:4] = np.flip(u_ns, axis=(0, 1))[:, :]
        u_a[4:7, 4:7] = u_ns[:, :]
        # Apply pi phase shifts on mode 3
        u_a[:, 3] = -u_a[:, 3]

        # Define beam splitter action
        u_bs = np.identity(8, dtype=complex)
        u_bs[3, 3] = 1 / 2**0.5
        u_bs[4, 4] = 1 / 2**0.5
        u_bs[3, 4] = 1j / 2**0.5
        u_bs[4, 3] = 1j / 2**0.5

        # Define mode reconfiguration so qubits are on central 4 modes
        swaps = {2: 0, 0: 1, 1: 2, 5: 7, 7: 6, 6: 5}
        u_perm1 = permutation_mat_from_swaps_dict(swaps, 8)
        u_perm2 = np.conj(u_perm1.T)

        u_a = u_perm2 @ u_bs @ u_a @ u_bs @ u_perm1

        unitary = Unitary(u_a)
        unitary.herald(0, 0, 0)
        unitary.herald(1, 1, 1)
        unitary.herald(1, 6, 6)
        unitary.herald(0, 7, 7)

        super().__init__(4)
        self.add(unitary, 0, group=True, name="CZ Heralded")


class CNOT_Heralded(PhotonicCircuit):  # noqa: N801
    """
    Heralded version of the CNOT gate which acts across two dual-rail encoded
    qubits, using two NS gates with ancillary photons to herald the success of
    the transformation. This gate occupies 8 modes, where modes 0 & 7 are used
    as 0 photon heralds, modes 1 & 6 are used as 1 photon heralds, mode 2 & 3
    correspond to the 0 & 1 states of the first qubit and modes 4 & 5 correspond
    to the 0 & 1 states of the second qubit. The heralded gate does not require
    any post-selection on the output qubits, other than that they are not lost
    (i.e a total of 4 photons should be measured at the output of the system),
    allowing it to be cascaded with other two qubit gates.

    Args:

        target_qubit (int, optional) : Sets which of the two qubits is used as
            the target qubit for the gate. Should be either 0 or 1.

    """

    def __init__(self, target_qubit: int = 1) -> None:
        if target_qubit not in [0, 1]:
            raise ValueError(
                "target_qubit setting must have a value of either 0 or 1."
            )

        super().__init__(4)

        # Create CNOT from combination of H and CZ
        circ = PhotonicCircuit(4)
        circ.add(H(), 2 * target_qubit)
        circ.add(CZ_Heralded(), 0)
        circ.add(H(), 2 * target_qubit)

        name = f"CNOT Heralded ({1 - target_qubit}, {target_qubit})"
        self.add(circ, 0, group=True, name=name)


class SWAP(PhotonicCircuit):
    """
    Performs a swap between the modes encoding two qubits. To do this, it needs
    to be provided with two tuples, each detailing the two modes used to encode
    the qubit.

    Args:

        qubit_1 (tuple) : A tuple detailing the modes used to encode the 0 & 1
            modes of the first qubit. Should be of the form (m0, m1).

        qubit_2 (tuple) : A tuple detailing the modes used to encode the 0 & 1
            modes of the second qubit. Should be of the form (m0, m1).

    """

    def __init__(
        self, qubit_1: tuple[int, int], qubit_2: tuple[int, int]
    ) -> None:
        # Confirm supplied form is correct
        if len(qubit_1) != 2:
            raise ValueError(
                "qubit_1 value should be a tuple containing two integer mode "
                "numbers."
            )
        if len(qubit_2) != 2:
            raise ValueError(
                "qubit_2 value should be a tuple containing two integer mode "
                "numbers."
            )
        # Extract mode values
        a0, a1 = qubit_1
        b0, b1 = qubit_2
        # Check each mode is an integer
        modes = [a0, a1, b0, b1]
        for m in modes:
            if not isinstance(m, int) or isinstance(m, bool):
                raise TypeError(
                    "One or more mode numbers detected not to be integer."
                )
        n_modes = max(modes) + 1

        # Create circuit with required number of modes
        super().__init__(n_modes)

        # Add required swaps
        self.mode_swaps({a0: b0, b0: a0, a1: b1, b1: a1})
