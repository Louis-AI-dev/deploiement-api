import unittest
import numpy as np
from api import preprocess_and_predict

class APITestCase(unittest.TestCase):
    def test_prediction_format(self):
        result = preprocess_and_predict(["This is a test tweet"])
        self.assertTrue(isinstance(result, (list, tuple, np.ndarray)))

if __name__ == '__main__':
    unittest.main()