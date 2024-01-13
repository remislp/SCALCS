#! /usr/bin/env python

import unittest
from tests import test_unit as tt

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(tt.TestDC_PyPs)
    unittest.TextTestRunner(verbosity=2).run(suite)
