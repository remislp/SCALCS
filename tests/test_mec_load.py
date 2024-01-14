from samples import samples
from scalcs import scalcsio
from scalcs import scburst
from scalcs import scalcslib as scl
from scalcs import qmatlib as qml

import os
import sys
import time
import unittest
import numpy as np


class TestMecLoad(unittest.TestCase):

    def setUp(self):
        filename = './samples/samples.xlsx'
        os.path.isfile(filename)
        self.mec = scalcsio.load_from_excel_sheet(filename, sheet=0)
        self.conc = 100e-9 # 100 nM
        self.mec.set_eff('c', self.conc)
        self.tres = 0.0001 # 100 microsec
        self.tcrit = 0.004

    def test_load_from_excel(self):
        assert len(self.mec.States) == 5
        assert len(self.mec.Rates) == 10
        assert len(self.mec.Cycles) == 1

    def test_burst(self):
        # # # Burst initial vector.
        phiB = scburst.phiBurst(self.mec)
        self.assertAlmostEqual(phiB[0], 0.275362, 6)
        self.assertAlmostEqual(phiB[1], 0.724638, 6)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMecLoad)
    unittest.TextTestRunner(verbosity=2).run(suite)

def test_mr():
    filename = './samples/samples.xlsx'
    mec = scalcsio.load_from_excel_sheet(filename, sheet=1)
    assert mec.Rates[8].rateconstants == 3.0
    assert mec.Rates[10].rateconstants == 2.0
    mec.update_mr()
    assert mec.Rates[8].rateconstants == 1.0
    assert mec.Rates[10].rateconstants == 1.0