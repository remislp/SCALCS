import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from samples import samples
from scalcs.qmatlib import eigenvalues_and_spectral_matrices, expQ, powQ 
from scalcs.qmatlib import pinf_extendQ, pinf_reduceQ

class TestQMatrixRegression(unittest.TestCase):

    def setUp(self):
        # Set up your QMatrix instance here
        # Example:
        self.mec = samples.CH82()
        self.mec.set_eff('c', 0.0000001)   
 
    def test_pinf_method1(self):
        # Test Q extention method to calculate Pinf
        result = pinf_extendQ(self.mec.Q)
        expected_result = np.array([2.48271431e-05, 0.00186203552, 0.00496542821, 6.20678511e-05, 0.993085641])
        np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_pinf_method2(self):
        # Test Q reduction method to calculate Pinf
        result = pinf_reduceQ(self.mec.Q)
        expected_result = np.array([2.48271431e-05, 0.00186203552, 0.00496542821, 6.20678511e-05, 0.993085641])
        np.testing.assert_allclose(result, expected_result, rtol=1e-6)


class TestEquilibriumOccupancies(unittest.TestCase):

    def test_pinf_extendQ(self):
        Q = np.array([[0, 1, -1],
                      [-1, 0, 1],
                      [1, -1, 0]])
        expected_pinf = np.array([0.33333333, 0.33333333, 0.33333333])
        pinf = pinf_extendQ(Q)
        assert_almost_equal(pinf, expected_pinf, decimal=5)

    def test_pinf_reduceQ(self):
        Q = np.array([[0, 1, -1],
                      [-1, 0, 1],
                      [1, -1, 0]])
        expected_pinf = np.array([0.33333333, 0.33333333, 0.33333333])
        pinf = pinf_reduceQ(Q)
        assert_almost_equal(pinf, expected_pinf, decimal=5)

class TestMatrixFunctions(unittest.TestCase):

    def test_eigenvalues_and_spectral_matrices(self):
        Q = np.array([[1, 2], [3, 4]])
        eigvals, A = eigenvalues_and_spectral_matrices(Q)
        
        expected_eigvals, expected_vectors = np.linalg.eig(Q)
        expected_inv_vectors = np.linalg.inv(expected_vectors)
        expected_A = np.einsum('ij,kj->kij', expected_vectors, expected_inv_vectors)
        
        sorted_indices = expected_eigvals.real.argsort()
        expected_eigvals = expected_eigvals[sorted_indices]
        expected_A = expected_A[sorted_indices]

        assert_almost_equal(eigvals, expected_eigvals)
        #assert_almost_equal(A, expected_A)

#    def test_expQ(self):
#        Q = np.array([[0, -1], [1, 0]])
#        t = np.pi
#        result = expQ(Q, t)
#        expected = np.array([[-1, 0], [0, -1]])
#        np.testing.assert_allclose(result, expected, rtol=1e-6)
#        #assert_almost_equal(result, expected)

    def test_powQ(self):
        Q = np.array([[2, 0], [0, 3]])
        n = 3
        result = powQ(Q, n)
        expected = np.array([[8, 0], [0, 27]])

        assert_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
