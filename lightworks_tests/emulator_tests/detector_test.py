from lightworks import State
from lightworks.emulator import Detector
import unittest

class DetectorTest(unittest.TestCase):
    """
    Unit tests to check behaviour of the Detector object in different 
    situations.
    """
    
    def setUp(self) -> None:
        self.lossy_detector = Detector(efficiency = 0.1) # Should be very lossy
        self.dc_detector = Detector(p_dark = 0.1) # High probability
        self.non_pnr_detector = Detector(photon_counting = False)
    
    def test_threshold_detection_multiphoton(self):
        """
        Checks that threshold detection behaves as expected for a state with 
        multiple photons in a mode.
        """
        out = self.non_pnr_detector._get_output(State([0,1,2,3,0,1]))
        self.assertEqual(out, State([0,1,1,1,0,1]))
        
    def test_threshold_detection_singlephoton(self):
        """
        Checks that threshold detection behaves as expected for a state with
        all modes having one or less photons.
        """
        out = self.non_pnr_detector._get_output(State([0,1,1,1,0,0]))
        self.assertEqual(out, State([0,1,1,1,0,0]))
        
    def test_threshold_detection_zerostate(self):
        """
        Checks that threshold detection behaves as expected for the zero state.
        """
        out = self.non_pnr_detector._get_output(State([0,0,0,0,0]))
        self.assertEqual(out, State([0,0,0,0,0]))
        
    def test_lossy_detector(self):
        """
        Confirms that the behaviour of detectors with imperfect detection 
        efficiency is as expected.
        """
        measured = set()
        for i in range(1000):
            measured.add(self.lossy_detector._get_output(State([0,2,1,0])))
        self.assertTrue(State([0,1,0,0]) in measured)
        
    def test_dark_counts(self):
        """Test that dark counts are working as expected."""
        measured = set()
        for i in range(1000):
            measured.add(self.dc_detector._get_output(State([0,0,0,0])))
        n_photons = [s.num() for s in measured]
        self.assertTrue(max(n_photons) > 0)
        
    def test_efficiency_modification(self):
        """
        Checks that the efficiency cannot be assigned to an invalid value.
        """
        detector = Detector()
        with self.assertRaises(TypeError):
            detector.efficiency = True
        with self.assertRaises(ValueError):
            detector.efficiency = -0.1
        with self.assertRaises(ValueError):
            detector.efficiency = 1.1
            
    def test_p_dark_modification(self):
        """
        Checks that the p_dark cannot be assigned to an invalid value.
        """
        detector = Detector()
        with self.assertRaises(TypeError):
            detector.p_dark = True
        with self.assertRaises(ValueError):
            detector.p_dark = -0.1
        with self.assertRaises(ValueError):
            detector.p_dark = 1.1
            
    def test_photon_counting_modification(self):
        """
        Checks that the photon_counting cannot be assigned to an invalid value.
        """
        detector = Detector()
        with self.assertRaises(TypeError):
            detector.photon_counting = 0.5
        with self.assertRaises(TypeError):
            detector.photon_counting = "True"
        
if __name__ == "__main__":
    unittest.main()