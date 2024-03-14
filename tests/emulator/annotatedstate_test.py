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

from lightworks.emulator.annotated_state import AnnotatedState
from lightworks.emulator import AnnotatedStateError

import pytest

class TestAnnotatedState:
    """Unit tests for AnnotatedState object."""
    
    def test_get_mode_value(self):
        """Confirms that mode value retrieval works as expected."""
        s = AnnotatedState([[],[0],[1],[]])
        assert s[1] == [0]
        
    def test_state_equality(self):
        """Checks that state equality comparison works as expected."""
        assert AnnotatedState([[],[0],[1],[]]) ==  \
                        AnnotatedState([[],[0],[1],[]])
        assert not AnnotatedState([[],[0],[1],[]]) ==  \
                         AnnotatedState([[],[1],[0],[]])
    
    def test_state_equality_shuffled(self):
        """
        Checks that state equality comparison works as expected in the case 
        that identical labels are provide in a mode but in a different order.
        """
        assert AnnotatedState([[],[0],[1,2],[]]) ==  \
                        AnnotatedState([[],[0],[2,1],[]])
        
    def test_state_addition(self):
        """Checks that state addition behaviour is as expected."""
        s = AnnotatedState([[0],[2,3],[1]]) + AnnotatedState([[],[0,3],[1]])
        assert s == AnnotatedState([[0],[2,3],[1],[],[0,3],[1]])
        s = AnnotatedState([[0],[2,3],[1]]) + AnnotatedState([])
        assert s == AnnotatedState([[0],[2,3],[1]])
        
    def test_merge(self):
        """Checks that state merge works correctly."""
        s = AnnotatedState([[0],[2,3],[1]]).merge(AnnotatedState([[],[0],[1]]))
        assert s == AnnotatedState([[0],[2,3,0],[1,1]])
        
    def test_modification_behavior(self):
        """
        Checks that the correct error is raised if we try to modify the State 
        value. Also tests what happens when state s attribute is modified.
        """
        s = AnnotatedState([[0,1],[2]])
        with pytest.raises(AnnotatedStateError):
            s.s = [[0]]
        s.s[0] = [2]
        assert s == AnnotatedState([[0,1],[2]])
        
    def test_mode_length(self):
        """Check the calculated mode number attribute is set correctly."""
        assert len(AnnotatedState([[],[0],[1,2],[],[0]])) == 5
        assert len(AnnotatedState([[],[],[],[]])) == 4
        assert len(AnnotatedState([])) == 0
        
    def test_photon_number(self):
        """Checks calculated photon number value is correct."""
        assert AnnotatedState([[],[0],[1,2],[],[0]]).n_photons == 4
        assert AnnotatedState([[],[],[],[]]).n_photons == 0
        assert AnnotatedState([]).n_photons == 0
