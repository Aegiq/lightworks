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

from lightworks import Parameter, random_unitary
from lightworks.sdk.circuit.photonic_compiler import CompiledPhotonicCircuit
from lightworks.sdk.circuit.photonic_components import (
    Barrier,
    BeamSplitter,
    Group,
    Loss,
    ModeSwaps,
    PhaseShifter,
    UnitaryMatrix,
)
from lightworks.sdk.utils import ModeRangeError


class TestCompiledCircuit:
    """
    Unit tests to confirm correct functioning of the CompiledPhotonicCircuit
    class when various operations are performed.
    """

    def test_creation(self):
        """
        Checks creation of a compiled circuit doesn't raise any issues.
        """
        CompiledPhotonicCircuit(10)

    def test_creation_modes(self):
        """
        Checks number of modes are correct on creation of a circuit.
        """
        c = CompiledPhotonicCircuit(10)
        assert c.n_modes == 10
        assert c.loss_modes == 00
        assert c.n_modes == c.total_modes

    def test_default_array_type(self):
        """
        Checks that default array type is complex.
        """
        c = CompiledPhotonicCircuit(10)
        assert c.U_full.dtype == complex

    @pytest.mark.parametrize(
        "component",
        [
            Barrier(list(range(4))),
            BeamSplitter(1, 3, 0.4, "Rx"),
            Loss(1, 0.5),
            ModeSwaps({0: 2, 2: 1, 1: 0}),
            PhaseShifter(3, 0.6),
            UnitaryMatrix(1, random_unitary(4), ""),
        ],
    )
    def test_component_addition(self, component):
        """
        Confirms each component can be added to a circuit and the configured
        unitary matches that from the component.
        """
        c = CompiledPhotonicCircuit(5)
        c.add(component)
        assert (
            c.U_full.round(8) == component.get_unitary(c.total_modes).round(8)
        ).all()

    def test_component_addition_group(self):
        """
        Confirms a group can be added to the circuit and that the unitary
        matches that which is manually calculated.
        """
        spec = [
            BeamSplitter(1, 2, 0.6, "Rx"),
            PhaseShifter(3, 0.6),
            Loss(1, 0.5),
            ModeSwaps({0: 2, 2: 1, 1: 0}),
        ]
        group = Group(spec, "test", 1, 3, {})
        c = CompiledPhotonicCircuit(5)
        c.add(group)
        # Find expected unitary from components individually
        expected = np.identity(c.total_modes)
        for s in spec:
            expected = s.get_unitary(c.total_modes) @ expected
        assert (c.U_full.round(8) == expected.round(8)).all()

    def test_parameter_support(self):
        """
        Confirms a parameter can be assigned to a component and it changes the
        produced circuit.
        """
        param = Parameter(0.9)
        bs = BeamSplitter(1, 3, param, "H")
        # Get first unitary
        c = CompiledPhotonicCircuit(4)
        c.add(bs)
        u1 = c.U_full
        # Then update parameter and get second
        param.set(0.3)
        c = CompiledPhotonicCircuit(4)
        c.add(bs)
        u2 = c.U_full
        assert (u1.round(8) != u2.round(8)).any()

    def test_herald(self):
        """
        Confirms that heralding being added to a circuit works as expected and
        is reflected in the heralds attribute.
        """
        circuit = CompiledPhotonicCircuit(4)
        circuit.add_herald(1, 0, 2)
        # Check heralds added
        assert 0 in circuit.heralds["input"]
        assert 2 in circuit.heralds["output"]
        # Check photon number is correct
        assert circuit.heralds["input"][0] == 1
        assert circuit.heralds["output"][2] == 1

    def test_herald_single_value(self):
        """
        Confirms that heralding being added to a circuit works as expected and
        is reflected in the heralds attribute, when only a single mode is set
        """
        circuit = CompiledPhotonicCircuit(4)
        circuit.add_herald(2, 1)
        # Check heralds added
        assert 1 in circuit.heralds["input"]
        assert 1 in circuit.heralds["output"]
        # Check photon number is correct
        assert circuit.heralds["input"][1] == 2
        assert circuit.heralds["output"][1] == 2

    @pytest.mark.parametrize("value", [2.5, "2", True])
    def test_herald_invalid_photon_number(self, value):
        """
        Checks error is raised when a non-integer photon number is provided to
        the herald method.
        """
        circuit = CompiledPhotonicCircuit(4)
        with pytest.raises(TypeError):
            circuit.add_herald(value, 0)

    @pytest.mark.parametrize("value", [2.5, 6, True])
    def test_herald_invalid_mode_number(self, value):
        """
        Checks error is raised when a non-integer mode number is provided to
        the herald method.
        """
        circuit = CompiledPhotonicCircuit(4)
        with pytest.raises((TypeError, ModeRangeError)):
            circuit.add_herald(0, value)

    @pytest.mark.parametrize("value", [2.5, 6, True])
    def test_herald_invalid_mode_number_output(self, value):
        """
        Checks error is raised when a non-integer output mode number is provided
        to the herald method.
        """
        circuit = CompiledPhotonicCircuit(4)
        with pytest.raises((TypeError, ModeRangeError)):
            circuit.add_herald(0, 0, value)

    def test_herald_duplicate_value(self):
        """
        Checks error is raised if a duplicate mode is provided to the herald.
        """
        circuit = CompiledPhotonicCircuit(4)
        circuit.add_herald(0, 0)
        with pytest.raises(ValueError):
            circuit.add_herald(0, 0)

    def test_herald_duplicate_value_output(self):
        """
        Checks error is raised if a duplicate output mode is provided to the
        herald.
        """
        circuit = CompiledPhotonicCircuit(4)
        circuit.add_herald(0, 0, 1)
        with pytest.raises(ValueError):
            circuit.add_herald(0, 1, 1)
