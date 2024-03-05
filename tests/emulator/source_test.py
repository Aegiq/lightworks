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

from lightworks import State
from lightworks.emulator.annotated_state import AnnotatedState
from lightworks.emulator import Source
import unittest

class SourceTest(unittest.TestCase):
    """
    Unit tests to check behaviour and statistics returned while using the 
    Source object of the emulator.
    """
    
    def test_perfect_input(self):
        """Checks only one state is returned when a perfect input is used."""
        source = Source()
        stats = source._build_statistics(State([1,0,1,0]))
        self.assertDictEqual(stats, {State([1,0,1,0]) : 1})
        
    def test_imperfect_brightness(self):
        """Checks the return values when an imperfect brightness is used."""
        source = Source(brightness = 0.915)
        stats = source._build_statistics(State([1,0,1,0]))
        stats = {s : round(p, 6) for s, p in stats.items()}
        expected = {State([1,0,1,0]): 0.837225, State([1,0,0,0]): 0.077775,
                    State([0,0,1,0]): 0.077775, State([0,0,0,0]): 0.007225}
        self.assertDictEqual(stats, expected)
        
    def test_imperfect_source(self):
        """
        Checks return values are correct when all possible imperfections in a 
        source are used. Also, confirms all states returned are of the correct 
        length.
        """
        source = Source(brightness = 0.493, purity = 0.984, 
                        indistinguishability = 0.937)
        stats = source._build_statistics(State([0,0,2,1,0,1,0,0]))
        # Check number of inputs is correct
        self.assertEqual(len(stats), 204)
        # Check some random values
        self.assertAlmostEqual(
            stats[AnnotatedState([[],[],[0],[0],[],[0],[],[]])], 
            0.10844531898317525, 6
            )
        self.assertAlmostEqual(stats[AnnotatedState([[]]*8)], 
                               0.06502113537379095, 6)
        # Ensure all state lengths are correct
        for state in stats:
            self.assertEqual(len(state), 8)
            
    def test_source_thresholding(self):
        """
        Confirms correct behaviour when probability thresholding is applied to 
        the Source object.
        """
        source = Source(brightness = 0.493, purity = 0.984, 
                        indistinguishability = 0.937, 
                        probability_threshold = 1e-6)
        stats = source._build_statistics(State([0,0,2,1,0,1,0,0]))
        # Check length is as expected after thresholding
        self.assertEqual(len(stats), 89)
        
    def test_purity_modification(self):
        """
        Checks that the purity cannot be assigned to an invalid value.
        """
        source = Source()
        with self.assertRaises(TypeError):
            source.purity = True
        with self.assertRaises(ValueError):
            source.purity = 0.5
        with self.assertRaises(ValueError):
            source.purity = 1.1
            
    def test_brightness_modification(self):
        """
        Checks that the brightness cannot be assigned to an invalid value.
        """
        source = Source()
        with self.assertRaises(TypeError):
            source.brightness = True
        with self.assertRaises(ValueError):
            source.brightness = -0.1
        with self.assertRaises(ValueError):
            source.brightness = 1.1
            
    def test_indistinguishability_modification(self):
        """
        Checks that the indistinguishability cannot be assigned to an invalid 
        value.
        """
        source = Source()
        with self.assertRaises(TypeError):
            source.indistinguishability = True
        with self.assertRaises(ValueError):
            source.indistinguishability = -0.1
        with self.assertRaises(ValueError):
            source.indistinguishability = 1.1
            
    def test_probability_threshold_modification(self):
        """
        Checks that the probability_threshold cannot be assigned to an invalid 
        value.
        """
        source = Source()
        with self.assertRaises(TypeError):
            source.probability_threshold = True
        with self.assertRaises(ValueError):
            source.probability_threshold = -0.1
        with self.assertRaises(ValueError):
            source.probability_threshold = 1.1
        
if __name__ == "__main__":
    unittest.main()