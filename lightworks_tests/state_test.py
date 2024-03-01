from lightworks import State
from lightworks.sdk.utils import StateError
import unittest

class StateTest(unittest.TestCase):
    """Unit tests for State object."""
    
    def test_get_mode_value(self):
        """Confirms that mode value retrieval works as expected."""
        s = State([1,2,3])
        self.assertEqual(s[1], 2)
        
    def test_state_equality(self):
        """Checks that state equality comparison works as expected."""
        self.assertTrue(State([1,0,1,0]) == State([1,0,1,0]))
        self.assertFalse(State([1,0,1,0]) == State([0,1,1,0]))
        
    def test_state_addition(self):
        """Checks that state addition behaviour is as expected."""
        s = State([1,2,3]) + State([0,4,5])
        self.assertEqual(s, State([1,2,3,0,4,5]))
        s = State([1,2,3]) + State([])
        self.assertEqual(s, State([1,2,3]))
        
    def test_merge(self):
        """Checks that state merge works correctly."""
        s = State([1,2,3]).merge(State([0,2,1]))
        self.assertEqual(s, State([1,4,4]))
        
    def test_modification_behavior(self):
        """
        Checks that the correct error is raised if we try to modify the State 
        value and checks behaviour if the s attribute of the state is modified.
        """
        s = State([1,2,3])
        with self.assertRaises(StateError):
            s.s = [1]
        s.s[0] = 2
        self.assertEqual(s, State([1,2,3]))
        
    def test_mode_length(self):
        """Check the calculated mode number attribute is set correctly."""
        self.assertEqual(len(State([1,2,0,1,0])), 5)
        self.assertEqual(len(State([0,0,0,0])), 4)
        self.assertEqual(len(State([])), 0)
        
    def test_photon_number(self):
        """Checks calculated photon number value is correct."""
        self.assertEqual(State([1,0,2,1,1]).num(), 5)
        self.assertEqual(State([0,0,0,0,0]).num(), 0)
        self.assertEqual(State([]).num(), 0)
        
if __name__ == "__main__":
    unittest.main()