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

from lightworks.sdk.circuit.circuit_compiler import CompiledCircuit

import pytest
from numpy import round

class TestCompiledCircuit:
    """
    Unit tests to confirm correct functioning of the CompiledCircuit class when
    various operations are performed.
    """
    
    def test_circuit_addition(self):
        """
        Compares a circuit created all at once to two circuits added together
        and checks they are equivalent.
        """
        # Comparison circuit
        circ_comp = CompiledCircuit(4)
        # First part
        for i, m in enumerate([0,2,0,1,0]):
            circ_comp.add_bs(m)
            circ_comp.add_ps(m, i)
        # Second part
        for i, m in enumerate([2,0,2,1,0]):
            circ_comp.add_ps(m+1, i)
            circ_comp.add_bs(m)
            circ_comp.add_ps(m, i)
        # Addition circuit
        c1 = CompiledCircuit(4)
        for i, m in enumerate([0,2,0,1,0]):
            c1.add_bs(m)
            c1.add_ps(m, i)
        c2 = CompiledCircuit(4)
        for i, m in enumerate([2,0,2,1,0]):
            c2.add_ps(m+1, i)
            c2.add_bs(m)
            c2.add_ps(m, i)
        circ_add = c1 + c2
        for i in range(circ_comp.U_full.shape[0]):
            for j in range(circ_comp.U_full.shape[1]):
                assert (circ_comp.U_full[i,j] == 
                        pytest.approx(circ_add.U_full[i,j], 1e-6))
                
    def test_two_lossy_circuit_addition(self):
        """
        Compares a circuit created all at once to two circuits added together
        and checks they are equivalent, with the addition of loss modes.
        """
        # Comparison circuit
        circ_comp = CompiledCircuit(4)
        # First part
        for i, m in enumerate([0,2,0,1,0]):
            circ_comp.add_bs(m)
            circ_comp.add_loss(m, 0.3*i)
            circ_comp.add_ps(m, i)
        # Second part
        for i, m in enumerate([2,0,2,1,0]):
            circ_comp.add_ps(m+1, i)
            circ_comp.add_bs(m)
            circ_comp.add_loss(m, 0.2*i)
            circ_comp.add_ps(m, i)
        # Addition circuit
        c1 = CompiledCircuit(4)
        for i, m in enumerate([0,2,0,1,0]):
            c1.add_bs(m)
            c1.add_loss(m, 0.3*i)
            c1.add_ps(m, i)
        c2 = CompiledCircuit(4)
        for i, m in enumerate([2,0,2,1,0]):
            c2.add_ps(m+1, i)
            c2.add_bs(m)
            c2.add_loss(m, 0.2*i)
            c2.add_ps(m, i)
        circ_add = c1 + c2
        for i in range(circ_comp.U_full.shape[0]):
            for j in range(circ_comp.U_full.shape[1]):
                assert (circ_comp.U_full[i,j] == 
                        pytest.approx(circ_add.U_full[i,j], 1e-6))
                
    def test_one_lossy_circuit_addition(self):
        """
        Compares a circuit created all at once to two circuits added together 
        and checks they are equivalent, with the addition of loss modes on the
        circuit which is to be added.
        """
        # Comparison circuit
        circ_comp = CompiledCircuit(4)
        # First part
        for i, m in enumerate([0,2,0,1,0]):
            circ_comp.add_bs(m)
            circ_comp.add_ps(m, i)
        # Second part
        for i, m in enumerate([2,0,2,1,0]):
            circ_comp.add_ps(m+1, i)
            circ_comp.add_bs(m)
            circ_comp.add_ps(m, i)
            circ_comp.add_loss(m, 0.1)
        # Addition circuit
        c1 = CompiledCircuit(4)
        for i, m in enumerate([0,2,0,1,0]):
            c1.add_bs(m)
            c1.add_ps(m, i)
        c2 = CompiledCircuit(4)
        for i, m in enumerate([2,0,2,1,0]):
            c2.add_ps(m+1, i)
            c2.add_bs(m)
            c2.add_ps(m, i)
            c2.add_loss(m, 0.1)
        circ_add = c1 + c2
        for i in range(circ_comp.U_full.shape[0]):
            for j in range(circ_comp.U_full.shape[1]):
                assert (circ_comp.U_full[i,j] == 
                        pytest.approx(circ_add.U_full[i,j], 1e-6))
                
    def test_smaller_circuit_addition(self):
        """
        Confirms equivalence between building a single circuit and added a 
        larger circuit to a smaller one with the add method.
        """
        # Comparison circuit
        circ_comp = CompiledCircuit(6)
        # First part
        for i, m in enumerate([0,2,4,1,3,2]):
            circ_comp.add_bs(m)
            circ_comp.add_ps(m, i)
        # Second part
        for i, m in enumerate([3,1,3,2,1]):
            circ_comp.add_ps(m+1, i)
            circ_comp.add_bs(m)
            circ_comp.add_ps(m, i)
            circ_comp.add_loss(m, 0.1)
        # Addition circuit
        c1 = CompiledCircuit(6)
        for i, m in enumerate([0,2,4,1,3,2]):
            c1.add_bs(m)
            c1.add_ps(m, i)
        c2 = CompiledCircuit(4)
        for i, m in enumerate([2,0,2,1,0]):
            c2.add_ps(m+1, i)
            c2.add_bs(m)
            c2.add_ps(m, i)
            c2.add_loss(m, 0.1)
        c1.add(c2, 1)
        # Check unitary equivalence
        U1 = round(circ_comp.U_full, 8)
        U2 = round(c1.U_full, 8)
        assert (U1 == U2).all()
                
    def test_mode_modification(self):
        """
        Checks that the number of modes attribute cannot be modified as this is
        not intended for the circuit and could cause issues.
        """
        circuit = CompiledCircuit(4)
        with pytest.raises(AttributeError):
            circuit.n_modes = 6