import unittest
import numpy as np
from ninda2 import ahp

criteria_ratio_matrix = np.array([
  [1.00, 2.00, 3.00, 5.00, 7.00, 7.00],
  [0.50, 1.00, 3.00, 5.00, 7.00, 7.00],
  [0.33, 0.33, 1.00, 4.00, 5.00, 3.00],
  [0.2, 0.2, 0.25, 1.0, 3.0, 3.0 ],
  [0.14, 0.14, 0.20, 0.33, 1.0, 2.0],
  [0.14, 0.14, 0.33, 0.33, 0.5, 1.0]
])

class TestAhp(unittest.TestCase):

    def test_normalize(self):
        expected = np.array([
          [ 0.43, 0.52, 0.39, 0.32, 0.30, 0.30, 2.26, 0.37695 ],
          [ 0.22, 0.26, 0.39, 0.32, 0.30, 0.30, 1.78, 0.29738 ],
          [ 0.14, 0.09, 0.13, 0.26, 0.21, 0.13, 0.96, 0.15967 ],
          [ 0.09, 0.05, 0.03, 0.06, 0.13, 0.13, 0.49, 0.08211 ],
          [ 0.06, 0.04, 0.03, 0.02, 0.04, 0.09, 0.28, 0.04592 ],
          [ 0.06, 0.04, 0.04, 0.02, 0.02, 0.04, 0.23, 0.03798 ]
        ])
        actual = ahp.normalize(criteria_ratio_matrix=criteria_ratio_matrix, n=6, IR=1.24)
        result = np.allclose(expected, actual, atol=0.01)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()