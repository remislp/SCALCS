import unittest
import numpy as np

from samples import samples
from scalcs.scburst import SCBurst

class TestSCBurst(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup CH82 test data
        c = 0.0000001 # 0.1 uM
        mec = samples.CH82()
        mec.set_eff('c', c)
        cls.q_burst = SCBurst(mec.Q, mec.kA, mec.kB, mec.kC, mec.kD)

        cls.expected_start_burst = np.array([0.27536232, 0.72463768])
        cls.expected_end_burst = np.array([[0.9609028], [0.20595089]])
        cls.expected_mean_number_of_openings = 3.81864
        cls.expected_mean_burst_length = 7.3281
        cls.expected_mean_open_time = 7.16585
        cls.expected_mean_shut_time = 0.162258
        cls.expected_mean_shut_time_excl_single = 0.276814
        cls.expected_mean_shut_time_between_bursts = 3790.43

    def test_start_burst(self):
        result = self.q_burst.start_burst
        np.testing.assert_almost_equal(result, self.expected_start_burst, decimal=5)

    def test_end_burst(self):
        result = self.q_burst.end_burst
        np.testing.assert_almost_equal(result, self.expected_end_burst, decimal=5)

    def test_mean_number_of_openings(self):
        result = self.q_burst.mean_number_of_openings
        self.assertAlmostEqual(result, self.expected_mean_number_of_openings, places=5)

    def test_mean_burst_length(self):
        result = 1000 * self.q_burst.mean_length
        self.assertAlmostEqual(result, self.expected_mean_burst_length, places=4)

    def test_mean_open_time(self):
        result = 1000 * self.q_burst.mean_open_time
        self.assertAlmostEqual(result, self.expected_mean_open_time, places=4)

    def test_mean_shut_time(self):
        result = 1000 * self.q_burst.mean_shut_time
        self.assertAlmostEqual(result, self.expected_mean_shut_time, places=5)

    def test_mean_shut_time_excl_single(self):
        result = 1000 * self.q_burst.mean_shut_time / self.q_burst.probability_more_than_one_opening
        self.assertAlmostEqual(result, self.expected_mean_shut_time_excl_single, places=5)

    def test_mean_shut_time_between_bursts(self):
        result = 1000 * self.q_burst.mean_shut_times_between_bursts
        self.assertAlmostEqual(result, self.expected_mean_shut_time_between_bursts, places=2)

if __name__ == '__main__':
    unittest.main()
