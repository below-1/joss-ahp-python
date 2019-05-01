import unittest
import numpy as np
from ninda2 import vikor

alternatives = np.array([
  [1.00, 0.00, 4.00, 2.00, 0.00, 4.00],
  [1.00, 0.00, 3.00, 1.00, 0.00, 1.00],
  [4.00, 2.0, 4.0, 0.00, 3.00, 5.00],
  [2.00, 3.0, 4.0, 1.0, 2.0, 3.0 ],
  [4.00, 3.00, 4.00, 2.00, 2.00, 3.00],
  [4.00, 4.00, 4.00, 1.00, 1.00, 4.00]
])

class TestVikor(unittest.TestCase):

    def test_vikor(self):
        benefit_cols = [ True, False, True, True, True, True ]
        best, worst = vikor.best_worst(benefit_cols)
        norm = vikor.normalize(alternatives, best, worst)
        
        

if __name__ == '__main__':
    unittest.main()