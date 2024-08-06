import numpy as np
import numpy.linalg as nplin
import unittest

from scalcs.qmatlib import eigenvalues_and_spectral_matrices

class TestEigsFunction(unittest.TestCase):
    def setUp(self):
        self.Q = np.array([[1, 2], [3, 4]])
        self.eigvals_expected = np.array([-0.37228132, 5.37228132])
        self.spectral_matrices_shape = (2, 2, 2)

    def test_eigenvalues(self):
        eigvals, _ = eigenvalues_and_spectral_matrices(self.Q)
        np.testing.assert_allclose(eigvals, self.eigvals_expected, rtol=1e-5)

    def test_spectral_matrices_shape(self):
        _, A = eigenvalues_and_spectral_matrices(self.Q)
        self.assertEqual(A.shape, self.spectral_matrices_shape)

    def test_no_sorting(self):
        eigvals, A = eigenvalues_and_spectral_matrices(self.Q, do_sorting=False)
        self.assertEqual(len(eigvals), self.Q.shape[0])
        self.assertEqual(A.shape, self.spectral_matrices_shape)

    def test_sorting(self):
        eigvals, A = eigenvalues_and_spectral_matrices(self.Q, do_sorting=True)
        self.assertTrue(np.all(np.diff(eigvals.real) >= 0))
        self.assertEqual(A.shape, self.spectral_matrices_shape)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)