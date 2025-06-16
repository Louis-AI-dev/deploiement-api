# remake_preprocessor.py
from src.custom_transformers import CleanAndStemTweets
import joblib
import os
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

preprocessor = CleanAndStemTweets()

# simulate a pipeline si besoin
# from sklearn.pipeline import Pipeline
# preprocessor = Pipeline([("clean", CleanAndStemTweets())])

save_path = os.path.join("run_stem_LSTM_05-06-2025_12-16", "preprocessor.pkl")
joblib.dump(preprocessor, save_path)
print("✅ preprocessor.pkl recréé avec un chemin de module correct.")