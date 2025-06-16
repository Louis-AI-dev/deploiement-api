import unittest
#import numpy as np
from api import preprocess_and_predict
import nltk
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.custom_transformers import CleanAndStemTweets

# Initialiser les outils
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class APITestCase(unittest.TestCase):
    def test_prediction_format(self):
        result = preprocess_and_predict(["This is a test tweet"])
        self.assertTrue(isinstance(result, (list, tuple, np.ndarray)))

if __name__ == '__main__':
    unittest.main()