from lightworks.emulator.annotated_state import AnnotatedState
from lightworks.emulator import AnnotatedStateError
import unittest

class AnnotatedStateTest(unittest.TestCase):
    """Unit tests for AnnotatedState object."""
    
    def test_get_mode_value(self):
        """Confirms that mode value retrieval works as expected."""
        s = AnnotatedState([[],[0],[1],[]])
        self.assertEqual(s[1], [0])
        
    def test_state_equality(self):
        """Checks that state equality comparison works as expected."""
        self.assertTrue(AnnotatedState([[],[0],[1],[]]) == 
                        AnnotatedState([[],[0],[1],[]]))
        self.assertFalse(AnnotatedState([[],[0],[1],[]]) == 
                         AnnotatedState([[],[1],[0],[]]))
    
    def test_state_equality_shuffled(self):
        """
        Checks that state equality comparison works as expected in the case 
        that identical labels are provide in a mode but in a different order.
        """
        self.assertTrue(AnnotatedState([[],[0],[1,2],[]]) == 
                        AnnotatedState([[],[0],[2,1],[]]))
        
    def test_state_addition(self):
        """Checks that state addition behaviour is as expected."""
        s = AnnotatedState([[0],[2,3],[1]]) + AnnotatedState([[],[0,3],[1]])
        self.assertEqual(s, AnnotatedState([[0],[2,3],[1],[],[0,3],[1]]))
        s = AnnotatedState([[0],[2,3],[1]]) + AnnotatedState([])
        self.assertEqual(s, AnnotatedState([[0],[2,3],[1]]))
        
    def test_merge(self):
        """Checks that state merge works correctly."""
        s = AnnotatedState([[0],[2,3],[1]]).merge(AnnotatedState([[],[0],[1]]))
        self.assertEqual(s, AnnotatedState([[0],[2,3,0],[1,1]]))
        
    def test_modification_behavior(self):
        """
        Checks that the correct error is raised if we try to modify the State 
        value. Also tests what happens when state s attribute is modified.
        """
        s = AnnotatedState([[0,1],[2]])
        with self.assertRaises(AnnotatedStateError):
            s.s = [[0]]
        s.s[0] = [2]
        self.assertEqual(s, AnnotatedState([[0,1],[2]]))
        
    def test_mode_length(self):
        """Check the calculated mode number attribute is set correctly."""
        self.assertEqual(len(AnnotatedState([[],[0],[1,2],[],[0]])), 5)
        self.assertEqual(len(AnnotatedState([[],[],[],[]])), 4)
        self.assertEqual(len(AnnotatedState([])), 0)
        
    def test_photon_number(self):
        """Checks calculated photon number value is correct."""
        self.assertEqual(AnnotatedState([[],[0],[1,2],[],[0]]).num(), 4)
        self.assertEqual(AnnotatedState([[],[],[],[]]).num(), 0)
        self.assertEqual(AnnotatedState([]).num(), 0)
        
if __name__ == "__main__":
    unittest.main()