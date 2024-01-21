import numpy as np
import unittest

from samples import samples
from scalcs.scpulse import (
    first_latency_asymptotic_roots_areas,
    calculate_first_latency_pdf_exact,
    calculate_first_latency_pdf_asymptotic,
    calculate_first_latency_pdf_ideal,
    first_latency_exact_GAMAxx)


class TestFirstLatencyFunctions(unittest.TestCase):

    def setUp(self):
        self.tres = 0.0001
        c0, c1 = 0.0, 0.0001
        self.tseq = np.linspace(0, 0.001, 100)
        self.mec0 = samples.CH82()
        self.mec0.set_eff('c', c0)
        self.mec1 = samples.CH82()
        self.mec1.set_eff('c', c1)
        self.pinf = np.array([0.2, 0.3, 0.5])

    def test_first_latency_asymptotic_roots_areas(self):
        roots, areas = first_latency_asymptotic_roots_areas(self.tres, self.mec0, self.mec1)
        
        # Add assertions based on expected results
        self.assertIsInstance(roots, np.ndarray)
        self.assertIsInstance(areas, np.ndarray)
        
        self.assertEqual(len(roots), self.mec1.kF)
        self.assertEqual(len(areas), self.mec1.kF)

    def test_first_latency_exact_GAMAxx(self):
        eigen, gamma00, gamma10, gamma11 = first_latency_exact_GAMAxx(self.tres, self.mec1, self.pinf)

        # Add assertions based on expected results
        self.assertIsInstance(eigen, np.ndarray)
        self.assertIsInstance(gamma00, np.ndarray)
        self.assertIsInstance(gamma10, np.ndarray)
        self.assertIsInstance(gamma11, np.ndarray)

        self.assertEqual(len(eigen), self.mec1.k)
        self.assertEqual(len(gamma00), self.mec1.k)
        self.assertEqual(len(gamma10), self.mec1.k)
        self.assertEqual(len(gamma11), self.mec1.k)

    def test_calculate_first_latency_pdf_exact(self):
        epdf = calculate_first_latency_pdf_exact(self.tres, self.tseq, self.mec0, self.mec1)

        # Add assertions based on expected results
        self.assertIsInstance(epdf, np.ndarray)
        self.assertEqual(len(epdf), len(self.tseq))

    def test_calculate_first_latency_pdf_asymptotic(self):
        apdf = calculate_first_latency_pdf_asymptotic(self.tres, self.tseq, self.mec0, self.mec1)

        # Add assertions based on expected results
        self.assertIsInstance(apdf, np.ndarray)
        self.assertEqual(len(apdf), len(self.tseq))

    def test_calculate_first_latency_pdf_ideal(self):
        ipdf = calculate_first_latency_pdf_ideal(self.tseq, self.mec0, self.mec1)
        
        # Add assertions based on expected results
        self.assertIsInstance(ipdf, np.ndarray)
        self.assertEqual(len(ipdf), len(self.tseq))

if __name__ == '__main__':
    unittest.main()
