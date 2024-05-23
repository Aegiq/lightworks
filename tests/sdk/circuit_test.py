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

from lightworks import Parameter, ParameterDict, Circuit
from lightworks import Unitary, random_unitary
from lightworks.sdk.circuit.circuit_compiler import CompiledCircuit
from lightworks.sdk.circuit.circuit_compiler import CompiledUnitary
from lightworks import CircuitCompilationError

import pytest
from random import random, seed, randint
from numpy import round

class TestCircuit:
    """
    Unit tests to confirm correct functioning of the Circuit class when various 
    operations are performed.
    """
    
    def setup_method(self) -> None:
        """Create a circuit and associated parameters for testing."""
        N = 6
        self.param_circuit = Circuit(N)
        self.parameters = ParameterDict()
        self.original_parameters = ParameterDict()
        seed(1)
        for i in range(N-1):
            for j in range(i%2, N-i%2, 2):
                p1 = Parameter(random())
                p1c = Parameter(p1.get())
                p2 = Parameter(random())
                p2c = Parameter(p2.get())
                self.parameters[f"bs_{i}_{j}"] = p1
                self.parameters[f"ps_{i}_{j}"] = p2
                self.original_parameters[f"bs_{i}_{j}"] = p1c
                self.original_parameters[f"ps_{i}_{j}"] = p2c
                self.param_circuit.add_ps(j, self.parameters[f"ps_{i}_{j}"])
                self.param_circuit.add_bs(j)
                self.param_circuit.add_ps(j+1, self.parameters[f"bs_{i}_{j}"])
                self.param_circuit.add_bs(j, loss = 0.1)
                
    def test_resultant_unitary(self):
        """
        Checks that the resultant unitary from a parameterized circuit is as
        expected.
        """
        unitary = self.param_circuit.U
        assert unitary[0,0] == pytest.approx(0.1817783235792+0.261054657406j,1e-8)
        assert unitary[1,2] == pytest.approx(0.1094958407210-0.2882179078302j,1e-8)
        assert unitary[4,3] == pytest.approx(0.03978296812819+0.354080300183j,1e-8)
        
    def test_parameter_modification(self):
        """
        Confirms that parameter modification changes unitary in expected way.
        """
        self.parameters["bs_0_0"] = 4
        self.parameters["bs_0_2"] = 4
        unitary = self.param_circuit.U
        assert unitary[0,0] == pytest.approx(0.1382843851268-0.1276219199576j,1e-8)
        assert unitary[1,2] == pytest.approx(0.6893944687270+0.2987967171732j,1e-8)
        assert unitary[4,3] == pytest.approx(-0.82752490939-0.0051178352488j,1e-8)
        
    def test_circuit_addition(self):
        """Confirms two circuits are added together correctly."""
        new_circ = self.param_circuit + self.param_circuit
        unitary = new_circ.U
        assert unitary[0,0] == pytest.approx(0.2743757510982+0.6727464244294j,1e-8)
        assert unitary[1,2] == pytest.approx(-0.153884469732+0.0872489579891j,1e-8)
        assert unitary[4,3] == pytest.approx(-0.083445311860+0.154159863276j,1e-8)
        
    def test_smaller_circuit_addition(self):
        """
        Confirms equivalence between building a single circuit and added a 
        larger circuit to a smaller one with the add method.
        """
        # Comparison circuit
        circ_comp = Circuit(6)
        # First part
        for i, m in enumerate([0,2,4,1,3,2]):
            circ_comp.add_bs(m)
            circ_comp.add_ps(m, i)
        # Second part
        for i, m in enumerate([3,1,3,2,1]):
            circ_comp.add_ps(m+1, i)
            circ_comp.add_bs(m, loss = 0.2*i)
            circ_comp.add_ps(m, i, loss = 0.1)
        # Addition circuit
        c1 = Circuit(6)
        for i, m in enumerate([0,2,4,1,3,2]):
            c1.add_bs(m)
            c1.add_ps(m, i)
        c2 = Circuit(4)
        for i, m in enumerate([2,0,2,1,0]):
            c2.add_ps(m+1, i)
            c2.add_bs(m, loss = 0.2*i)
            c2.add_ps(m, i, loss = 0.1)
        c1.add(c2, 1)
        # Check unitary equivalence
        U1 = round(circ_comp.U_full, 8)
        U2 = round(c1.U_full, 8)
        assert (U1 == U2).all()
                
    def test_smaller_circuit_addition_grouped(self):
        """
        Confirms equivalence between building a single circuit and added a 
        larger circuit to a smaller one with the add method, while using the 
        group method.
        """
        # Comparison circuit
        circ_comp = Circuit(6)
        # First part
        for i, m in enumerate([0,2,4,1,3,2]):
            circ_comp.add_bs(m)
            circ_comp.add_ps(m, i)
        # Second part
        for i in range(4):
            for i, m in enumerate([3,1,3,2,1]):
                circ_comp.add_ps(m+1, i)
                circ_comp.add_bs(m, loss = 0.2*i)
                circ_comp.add_ps(m, i, loss = 0.1)
            circ_comp.add_mode_swaps({1:2, 2:3, 3:1})
        # Addition circuit
        c1 = Circuit(6)
        for i, m in enumerate([0,2,4,1,3,2]):
            c1.add_bs(m)
            c1.add_ps(m, i)
        c2 = Circuit(4)
        for i, m in enumerate([2,0,2,1,0]):
            c2.add_ps(m+1, i)
            c2.add_bs(m, loss = 0.2*i)
            c2.add_ps(m, i, loss = 0.1)
        c2.add_mode_swaps({0:1, 1:2, 2:0})
        # Test combinations of True and False for group option
        c2.add(c2, 0, group = True)
        c2.add(c2, 0, group = False)
        c1.add(c2, 1, group = True)
        # Check unitary equivalence
        U1 = round(circ_comp.U_full, 8)
        U2 = round(c1.U_full, 8)
        assert (U1 == U2).all()
        
    def test_barrier_inclusion(self):
        """
        Checks that barrier component can be added across all and a selected 
        mode range.
        """
        circuit = Circuit(4)
        circuit.add_barrier()
        circuit.add_barrier([0,2])
        
    def test_mode_not_parameter(self):
        """
        Checks that an error is raised if a parameter is attempted to be 
        assigned to a mode value.
        """
        new_circ = Circuit(4)
        with pytest.raises(TypeError):
            new_circ.add_bs(Parameter(1))
        with pytest.raises(TypeError):
            new_circ.add_ps(Parameter(1))
        with pytest.raises(TypeError):
            new_circ.add_mode_swaps({Parameter(1):2, 2:Parameter(1)})
            
    def test_circ_unitary_combination(self):
        """Test combination of a circuit and unitary objects."""
        circ = Circuit(6)
        for i, m in enumerate([0,2,4,1,3,2]):
            circ.add_bs(m, loss = 0.2)
            circ.add_ps(m, i)
        u1 = Unitary(random_unitary(6, seed = 1))
        u2 = Unitary(random_unitary(4, seed = 2))
        circ.add(u1, 0)
        circ.add(u2, 1)
        assert circ.U[0,0] == pytest.approx(0.2287112952348-0.14731470234581j,1e-8)
        assert circ.U[1,2] == pytest.approx(0.0474053983616+0.01248244201229j,1e-8)
        assert circ.U[4,3] == pytest.approx(0.0267553699139-0.02848937675632j,1e-8)
        
    def test_mode_modification(self):
        """
        Checks that n_modes attribute cannot be modified and will raise an 
        attribute error.
        """
        circ = Circuit(4)
        with pytest.raises(AttributeError):
            circ.n_modes = 6
        
    def test_circuit_copy(self):
        """Test copy method of circuit creates an independent circuit."""
        copied_circ = self.param_circuit.copy()
        U1 = self.param_circuit.U_full
        # Modify the new circuit and check the original U is unchanged
        copied_circ.add_bs(0)
        U2 = self.param_circuit.U_full
        assert (U1 == U2).all()
        
    def test_circuit_copy_parameter_modification(self):
        """Test parameter modification still works on a copied circuit"""
        copied_circ = self.param_circuit.copy()
        U1 = copied_circ.U_full
        # Modify parameter and get new unitary from copied circuit
        self.parameters["bs_0_0"] = 2
        U2 = copied_circ.U_full
        # Unitary should be modified
        assert not (U1 == U2).all()
        
    def test_circuit_copy_parameter_freeze(self):
        """
        Checks copy method of the circuit can be used with the freeze parameter
        argument to create a new circuit without the Parameter objects.
        """
        copied_circ = self.param_circuit.copy(freeze_parameters = True)
        U1 = copied_circ.U_full
        # Modify parameter and get new unitary from copied circuit
        self.parameters["bs_0_0"] = 4
        U2 = copied_circ.U_full
        # Unitary should not be modified
        assert (U1 == U2).all()
        
    def test_circuit_ungroup(self):
        """
        Check that the unpack_groups method removes any grouped components from
        the circuit.
        """
        # Create initial basic circuit
        circuit = Circuit(4)
        circuit.add_bs(0)
        circuit.add_bs(2)
        circuit.add_ps(1, 0)
        # Create smaller circuit to add and combine
        circuit2 = Circuit(2)
        circuit2.add_bs(0)
        circuit2.add_ps(1, 1)
        circuit2.add(circuit2, group = True)
        circuit.add(circuit2, 1, group = True)
        # Apply unpacking and check any groups have been removed
        circuit.unpack_groups()
        group_found = False
        for spec in circuit._get_circuit_spec():
            if spec[0] == "group":
                group_found = True
        assert not group_found
        
    def test_remove_non_adj_bs_success(self):
        """
        Checks that the remove_non_adjacent_bs method of the circuit is able
        to successfully remove all beam splitters which act on non-adjacent 
        modes.
        """
        # Create circuit with beam splitters across non-adjacent modes
        circuit = Circuit(8)
        circuit.add_bs(0, 7)
        circuit.add_bs(1, 4)
        circuit.add_bs(2, 6)
        circuit.add_bs(3, 7)
        circuit.add_bs(0, 1)
        # Apply method and check all remaining beam splitters
        circuit.remove_non_adjacent_bs()
        for spec in circuit._get_circuit_spec():
            if spec[0] == "bs":
                # Check it acts on adjacent modes, otherwise fail
                if spec[1][0] != spec[1][1] - 1:
                    pytest.fail()
        
    def test_remove_non_adj_bs_equivalence(self):
        """
        Checks that the remove_non_adjacent_bs method of the circuit retains 
        the circuit unitary.
        """
        # Create circuit with beam splitters across non-adjacent modes
        circuit = Circuit(8)
        circuit.add_bs(0, 7)
        circuit.add_bs(1, 4)
        circuit.add_bs(2, 6)
        circuit.add_bs(7, 3)
        circuit.add_bs(0, 1)
        # Apply method and check unitary equivalence
        u1 = abs(circuit.U).round(3)
        circuit.remove_non_adjacent_bs()
        u2 = abs(circuit.U).round(3)
        assert (u1 == u2).all()
                    
    def test_remove_non_adj_bs_equivalence_grouped(self):
        """
        Checks that the remove_non_adjacent_bs method of the circuit retains 
        the circuit unitary when grouped components are featured in the 
        circuit.
        """
        # Create circuit with beam splitters across non-adjacent modes
        circuit = Circuit(8)
        circuit.add_bs(0, 7)
        circuit.add_bs(1, 4)
        circuit.add_bs(2, 6)
        circuit.add_bs(3, 7)
        circuit.add_bs(0, 1)
        # Then create smaller second circuit and add, with group = True
        circuit2 = Circuit(6)
        circuit2.add_bs(1, 4)
        circuit2.add_bs(2, 5)
        circuit2.add_bs(3)
        circuit.add(circuit2, 1, group = True)
        # Apply method and check unitary equivalence
        u1 = abs(circuit.U).round(8)
        circuit.remove_non_adjacent_bs()
        u2 = abs(circuit.U).round(8)
        assert (u1 == u2).all()
    
    def test_compress_mode_swap_equivalance(self):
        """
        Tests the circuit compress_mode_swaps method retains the circuit 
        unitary.
        """
        # Create circuit with a few components and then mode swaps
        circuit = Circuit(8)
        circuit.add_bs(0)
        circuit.add_bs(4)
        circuit.add_mode_swaps({1:3, 3:5, 5:6, 6:1})
        circuit.add_mode_swaps({0:1, 2:4, 1:2, 4:0})
        circuit.add_mode_swaps({5:3, 3:5})
        circuit.add_bs(0)
        circuit.add_bs(4)
        # Apply method and check unitary equivalence
        u1 = abs(circuit.U).round(8)
        circuit.compress_mode_swaps()
        u2 = abs(circuit.U).round(8)
        assert (u1 == u2).all()
        
    def test_compress_mode_swap_removes_components(self):
        """
        Tests the circuit compress_mode_swaps method is able to reduce it down 
        to using 2 mode swaps for an example circuit.
        """
        # Create circuit with a few components and then mode swaps
        circuit = Circuit(8)
        circuit.add_bs(0)
        circuit.add_bs(4)
        circuit.add_mode_swaps({1:3, 3:5, 5:6, 6:1})
        circuit.add_mode_swaps({0:1, 2:4, 1:2, 4:0})
        circuit.add_mode_swaps({5:3, 3:5})
        circuit.add_bs(0)
        circuit.add_bs(4)
        circuit.add_mode_swaps({0:1, 2:4, 1:2, 4:0})
        # Apply method and check only two mode_swap components are present
        circuit.compress_mode_swaps()
        counter = 0
        for spec in circuit._get_circuit_spec():
            if spec[0] == "mode_swaps": counter += 1
        assert counter == 2
        
    def test_compress_mode_swap_ignores_groups(self):
        """Checks that the mode swap ignores components in groups."""
        # Create circuit with a few components and then two mode swaps, one 
        # placed within a group from a smaller circuit
        circuit = Circuit(8)
        circuit.add_bs(0)
        circuit.add_bs(4)
        circuit.add_mode_swaps({1:3, 3:5, 5:6, 6:1})
        circuit2 = Circuit(5)
        circuit2.add_mode_swaps({0:1, 2:4, 1:2, 4:0})
        circuit.add(circuit2, 1, group=True)
        circuit.add_bs(0)
        circuit.add_bs(4)
        # Apply method and check two mode_swap components are still present
        circuit.compress_mode_swaps()
        circuit.unpack_groups() # unpack groups for counting
        counter = 0
        for spec in circuit._get_circuit_spec():
            if spec[0] == "mode_swaps": counter += 1
        assert counter == 2
        
    def test_parameterized_loss(self):
        """Checks that loss can be parameterized in a circuit."""
        param = Parameter(0.5, label = "loss")
        circuit = Circuit(4)
        circuit.add_bs(0, loss = param)
        circuit.add_ps(2, 1, loss = param)
        circuit.add_loss(2, param)
        
    def test_add_herald(self):
        """
        Confirms that heralding being added to a circuit works as expected and 
        is reflected in the heralds attribute.
        """
        circuit = Circuit(4)
        circuit.add_herald(1, 0, 2)
        # Check heralds added
        assert 0 in circuit.heralds["input"]
        assert 2 in circuit.heralds["output"]
        # Check photon number is correct
        assert circuit.heralds["input"][0] == 1
        assert circuit.heralds["output"][2] == 1
        
    def test_add_herald_single_value(self):
        """
        Confirms that heralding being added to a circuit works as expected and 
        is reflected in the heralds attribute, when only a single mode is set
        """
        circuit = Circuit(4)
        circuit.add_herald(2, 1)
        # Check heralds added
        assert 1 in circuit.heralds["input"]
        assert 1 in circuit.heralds["output"]
        # Check photon number is correct
        assert circuit.heralds["input"][1] == 2
        assert circuit.heralds["output"][1] == 2
        
    @pytest.mark.parametrize("value", [2.5, "2", True])
    def test_add_herald_invalid_photon_number(self, value):
        """
        Checks error is raised when a non-integer photon number is provided to 
        add_herald.
        """
        circuit = Circuit(4)
        with pytest.raises(TypeError):
            circuit.add_herald(value)
    
    @pytest.mark.parametrize("modes", [[1],[2],[1,2]])
    def test_add_herald_duplicate(self, modes):
        """
        Checks error is raised when duplicate mode is added to herald.
        """
        circuit = Circuit(4)
        circuit.add_herald(1, 1, 2)
        with pytest.raises(ValueError):
            circuit.add_herald(1, *modes)
            
    def test_heralded_circuit_addition(self):
        """
        Checks that the heralds end up on the correct modes when a heralded 
        circuit is added to a larger circuit.
        """
        circuit = Circuit(4)
        # Create heralded sub-circuit to add
        sub_circ = Circuit(4)
        sub_circ.add_herald(1, 0, 0)
        sub_circ.add_herald(1, 3, 3)
        # Then add to larger circuit
        circuit.add(sub_circ, 1)
        # Check circuit size is increased
        assert circuit.n_modes == 6
        # Confirm heralds are on modes 1 and 4
        assert 1 in circuit.heralds["input"]
        assert 1 in circuit.heralds["output"]
        assert 4 in circuit.heralds["input"]
        assert 4 in circuit.heralds["output"]
        
    def test_heralded_circuit_addition_herald_modification(self):
        """
        Checks that heralds in an existing circuit are modified correctly when 
        a heralded sub-circuit is added.
        """
        # Place beam splitters across 0 & 1 and 2 & 3
        circuit = Circuit(4)
        circuit.add_herald(0, 0, 1)
        circuit.add_herald(2, 2, 3)
        # Create heralded sub-circuit and add
        sub_circ = Circuit(4)
        sub_circ.add_herald(1, 0, 0)
        sub_circ.add_herald(1, 3, 3)
        circuit.add(sub_circ, 1)
        # Check heralds are in correct locations
        assert 0 in circuit._external_heralds["input"]
        assert 2 in circuit._external_heralds["output"]
        assert 3 in circuit._external_heralds["input"]
        assert 5 in circuit._external_heralds["output"]
        
    def test_heralded_circuit_addition_circ_modification(self):
        """
        Checks that components in an existing circuit are modified correctly 
        when a heralded sub-circuit is added.
        """
        # Place beam splitters across 0 & 1 and 2 & 3
        circuit = Circuit(4)
        circuit.add_bs(0)
        circuit.add_bs(2)
        # Create heralded sub-circuit and add
        sub_circ = Circuit(4)
        sub_circ.add_herald(1, 0, 0)
        sub_circ.add_herald(1, 3, 3)
        circuit.add(sub_circ, 1)
        # Confirm beam splitters now act on modes 0 & 2 and 3 & 5
        spec = circuit._Circuit__circuit_spec
        # Get relevant elements from spec
        assert spec[0][1][0:2] == (0, 2)
        assert spec[1][1][0:2] == (3, 5)
        
    def test_heralded_circuit_addition_circ_modification_2(self):
        """
        Checks that components in an existing circuit are modified correctly 
        when a heralded sub-circuit is added in a different location.
        """
        # Place beam splitters across 0 & 1 and 2 & 3
        circuit = Circuit(4)
        circuit.add_bs(0)
        circuit.add_bs(2)
        # Create heralded sub-circuit and add
        sub_circ = Circuit(4)
        sub_circ.add_herald(1, 0, 0)
        sub_circ.add_herald(1, 3, 3)
        circuit.add(sub_circ, 0)
        # Confirm beam splitters now act on modes 0 & 2 and 3 & 5
        spec = circuit._Circuit__circuit_spec
        # Get relevant elements from spec
        assert spec[0][1][0:2] == (1, 2)
        assert spec[1][1][0:2] == (4, 5)
        
    def test_heralded_circuit_addition_circ_modification_3(self):
        """
        Checks that components in an existing circuit are modified correctly 
        when a heralded sub-circuit is added with a different configuration.
        """
        # Place beam splitters across 0 & 1 and 2 & 3
        circuit = Circuit(4)
        circuit.add_bs(0)
        circuit.add_bs(2)
        # Create heralded sub-circuit and add
        sub_circ = Circuit(4)
        sub_circ.add_herald(1, 0, 1)
        sub_circ.add_herald(1, 1, 0)
        circuit.add(sub_circ, 1)
        # Confirm beam splitters now act on modes 0 & 2 and 3 & 5
        spec = circuit._Circuit__circuit_spec
        # Get relevant elements from spec
        assert spec[0][1][0:2] == (0, 3)
        assert spec[1][1][0:2] == (4, 5)
        
    def test_heralded_circuit_addition_circ_modification_all_herald(self):
        """
        Checks that components in an existing circuit are modified correctly 
        when a heralded sub-circuit is added with all modes heralded.
        """
        # Place beam splitters across 0 & 1 and 2 & 3
        circuit = Circuit(4)
        circuit.add_bs(0)
        circuit.add_bs(2)
        # Create heralded sub-circuit and add
        sub_circ = Circuit(4)
        sub_circ.add_herald(1, 0, 0)
        sub_circ.add_herald(1, 1, 1)
        sub_circ.add_herald(1, 2, 2)
        sub_circ.add_herald(1, 3, 3)
        circuit.add(sub_circ, 1)
        # Confirm beam splitters now act on modes 0 & 2 and 3 & 5
        spec = circuit._Circuit__circuit_spec
        # Get relevant elements from spec
        assert spec[0][1][0:2] == (0, 5)
        assert spec[1][1][0:2] == (6, 7)
        
    def test_heralded_two_circuit_addition_circ_modification(self):
        """
        Checks that components in an existing circuit are modified correctly 
        when two heralded sub-circuits are added.
        """
        # Place beam splitters across 0 & 1 and 2 & 3
        circuit = Circuit(4)
        circuit.add_bs(0)
        circuit.add_bs(2)
        # Create heralded sub-circuit and add
        sub_circ = Circuit(4)
        sub_circ.add_herald(1, 0, 1)
        sub_circ.add_herald(1, 2, 0)
        circuit.add(sub_circ, 2)
        circuit.add(sub_circ, 1)
        # Confirm beam splitters now act on modes 0 & 2 and 3 & 5
        spec = circuit._Circuit__circuit_spec
        # Get relevant elements from spec
        assert spec[0][1][0:2] == (0, 2)
        assert spec[1][1][0:2] == (5, 7)
        
    def test_input_modes(self):
        """
        Checks input modes attribute returns n_modes value when no heralds are 
        used.
        """
        circuit = Circuit(randint(4, 10))
        assert circuit.input_modes == circuit.n_modes 
        
    def test_input_modes_heralds(self):
        """
        Checks input modes attribute returns less then n_modes value when 
        heralds are included.
        """
        n = randint(6, 10)
        circuit = Circuit(n)
        circuit.add_herald(1, 1, 4)
        circuit.add_herald(2, 3, 3)
        assert circuit.input_modes == n - 2 
        
    def test_input_modes_heralds_sub_circuit(self):
        """
        Checks input modes attribute returns original n_modes value when 
        heralds are included as part of a sub-circuit and then added to the 
        larger circuit.
        """
        n = randint(9, 10)
        circuit = Circuit(n)
        sub_circuit = Circuit(4)
        sub_circuit.add_herald(1, 0, 0)
        sub_circuit.add_herald(0, 3, 1)
        circuit.add(sub_circuit, 2)
        assert circuit.input_modes == n
        
    def test_input_modes_heralds_sub_circuit_original_heralds(self):
        """
        Checks input modes attribute returns original n_modes value when 
        heralds are included as part of a sub-circuit and then added to the 
        larger circuit, with the larger circuit also containing heralds.
        """
        n = randint(9, 10)
        circuit = Circuit(n)
        circuit.add_herald(1, 1, 4)
        circuit.add_herald(2, 3, 3)
        sub_circuit = Circuit(4)
        sub_circuit.add_herald(1, 0, 0)
        sub_circuit.add_herald(0, 3, 1)
        circuit.add(sub_circuit, 2)
        assert circuit.input_modes == n - 2
        
    @pytest.mark.parametrize("initial_modes,final_modes",[[(0,),(0,2)], 
                                                          [(2,),(5,6)],
                                                          [(2,3),(5,6)],
                                                          [(0,2),(0,5)]])
    def test_bs_correct_modes(self, initial_modes, final_modes):
        """
        Checks that when a circuit has internal modes then the beam splitter is
        applied correctly across the other modes. 
        """
        circuit = Circuit(7)
        circuit._Circuit__internal_modes = [1, 3, 4]
        # Add bs on modes
        circuit.add_bs(*initial_modes)
        # Then check modes are converted to correct values
        assert circuit._Circuit__circuit_spec[0][1][0] == final_modes[0]
        assert circuit._Circuit__circuit_spec[0][1][1] == final_modes[1]
        
    def test_add_bs_invalid_convention(self):
        """
        Checks a ValueError is raised if an invalid beam splitter convention is
        set in add_bs.
        """
        circuit = Circuit(3)
        with pytest.raises(ValueError):
            circuit.add_bs(0, convention = "Not valid")
            
    def test_get_all_params(self):
        """
        Tests get_all_params method correctly identifies all parameters added
        within a circuit.
        """
        p1 = Parameter(0.5)
        p2 = Parameter(2)
        p3 = Parameter(3)
        # Create circuit with parameters
        circuit = Circuit(4)
        circuit.add_bs(0, reflectivity = p1)
        circuit.add_ps(2, p2)
        circuit.add_loss(3, p3)
        # Recover all params and then check
        all_params = circuit.get_all_params()
        for param in [p1, p2, p3]:
            assert param in all_params
            
    def test_get_all_params_grouped_circ(self):
        """
        Tests get_all_params method correctly identifies all parameters added
        within a grouped circuit.
        """
        p1 = Parameter(0.5)
        p2 = Parameter(2)
        p3 = Parameter(3)
        # Create sub-circuit with parameters
        sub_circuit = Circuit(4)
        sub_circuit.add_bs(0, reflectivity = p1)
        sub_circuit.add_ps(2, p2)
        sub_circuit.add_loss(3, p3)
        # Then add to larger circuit
        circuit = Circuit(5)
        circuit.add(sub_circuit, 1, group = True)
        # Recover all params and then check
        all_params = circuit.get_all_params()
        for param in [p1, p2, p3]:
            assert param in all_params
        
    def test_edit_circuit_spec(self):
        """
        Checks an exception is raised if a circuit spec is modified with an 
        invalid value.
        """
        circuit = Circuit(4)
        circuit._Circuit__circuit_spec = [["test", None]]
        with pytest.raises(CircuitCompilationError):
            circuit._build()
        
class TestUnitary:
    """
    Unit tests to confirm correct functioning of the Unitary class when various 
    operations are performed.
    """
    
    def test_unitary_assignment(self):
        """Checks that a unitary is correctly assigned with the component."""
        u = random_unitary(4)
        unitary = Unitary(u)
        assert (u == unitary.U).all()
        
    def test_non_unitary_assignment(self):
        """
        Checks that errors are raised when non-unitary matrices are assigned.
        """
        # Non-square unitary
        u = random_unitary(4)
        u = u[:,:-2]
        with pytest.raises(ValueError):
            Unitary(u)
        # Non-unitary matrix
        u2 = random_unitary(4)
        u2[0,0] = 1
        with pytest.raises(ValueError):
            Unitary(u2)
            
    def test_circuit_addition_to_unitary(self):
        """
        Confirm that addition of a circuit to the Unitary object works as 
        expected.
        """
        u = Unitary(random_unitary(6, seed = 95))
        circ = Circuit(4)
        circ.add_bs(0)
        circ.add_bs(2, loss = 0.5)
        circ.add_bs(1, loss = 0.2)
        u.add(circ, 1)
        assert u.U[0,0] == pytest.approx(-0.27084817086493-0.176576418865914j,1e-8)
        assert u.U[1,2] == pytest.approx(0.232353190742325-0.444902420616067j,1e-8)
        assert u.U[4,3] == pytest.approx(-0.31290267006132-0.091957924939349j,1e-8)
        
    def test_unitary_is_circuit_child(self):
        """
        Checks that the unitary object is a child class of the Circuit object.
        """
        u = Unitary(random_unitary(4))
        assert isinstance(u, Circuit)
        
    def test_n_mode_retrival(self):
        """
        Confirms n_mode attribute retrieval works for unitary component.
        """
        u = Unitary(random_unitary(4))
        assert u.n_modes, 4