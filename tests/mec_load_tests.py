from scalcs.samples import samples
from scalcs import scalcsio
from scalcs import scburst
from scalcs import scalcslib as scl
from scalcs import qmatlib as qml

import os
import sys
import time
import unittest
import numpy as np


class TestDC_PyPs(unittest.TestCase):

    def setUp(self):
        self.mec = samples.CH82()
        self.conc = 100e-9 # 100 nM
        self.mec.set_eff('c', self.conc)
        self.tres = 0.0001 # 100 microsec
        self.tcrit = 0.004

    def test_burst(self):

        # # # Burst initial vector.
        phiB = scburst.phiBurst(self.mec)

        self.assertAlmostEqual(phiB[0], 0.275362, 6)
        self.assertAlmostEqual(phiB[1], 0.724638, 6)

    def test_load_from_excel(self):
        filename = './scalcs/samples/samples.xlsx'
        os.path.isfile(filename)
        print(os.path.abspath(os.getcwd()))
        mec = scalcsio.load_from_excel_sheet(filename, sheet=0)
        assert len(mec.States) == 5
        assert len(mec.Rates) == 10
        assert len(mec.Cycles) == 1

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDC_PyPs)
    unittest.TextTestRunner(verbosity=2).run(suite)