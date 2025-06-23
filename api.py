from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.custom_transformers import CleanAndStemTweets
import pickle

print("Début chargement modèle...")

# Configuration simple du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Test de log
logger.info("Logger initialisé. Prêt à logger dans Azure Log Stream 🎯")

# Initialiser l'application Flask
app = Flask(__name__)
    
# Chemin absolu vers le dossier de api.py
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

# Chemins vers les fichiers
model_path = os.path.join(BASE_DIR, "run_stem_LSTM_05-06-2025_12-16", "best_model.h5")
tokenizer_path = os.path.join(BASE_DIR, "run_stem_LSTM_05-06-2025_12-16", "tokenizer.pkl")
preprocessor_path = os.path.join(BASE_DIR, "run_stem_LSTM_05-06-2025_12-16", "preprocessor.pkl")

try:
    logger.info("Chargement du modèle TensorFlow...")
    model = tf.keras.models.load_model(model_path)
    logger.info("Chargement du tokenizer...")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    logger.info("Chargement du preprocessor...")
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Tous les artefacts chargés avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement des artefacts : {e}", exc_info=True)
    model = None
    tokenizer = None
    preprocessor = None

def preprocess_and_predict(texts):
    logger.info(f"Texte reçu: {texts}")
    cleaned = preprocessor.transform(texts)
    logger.info(f"Texte cleaned: {cleaned}")
    sequences = tokenizer.texts_to_sequences(cleaned)
    logger.info(f"Texte sequences: {sequences}")
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50, padding='post')
    logger.info(f"Texte pad: {padded}")
    proba = model.predict(padded)
    predictions = (proba >= 0.5).astype(int)  # convertit les probabilités en 0 ou 1
    logger.info(f"Prédictions = {predictions}")
    return predictions

@app.route('/')
def hello():
    return 'Hello depuis Azure !'

# Endpoint /predict
@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Requête reçue pour /predict")
    if model is None or tokenizer is None or preprocessor is None:
        logger.error("Modèle ou artefacts non chargés")
        return jsonify({'error': 'Le modèle ou les artefacts ne sont pas disponibles'}), 500

    try:
        data = request.get_json()
        logger.info(f"Données reçues : {data}")

        if "text" not in data:
            logger.error("Champ 'text' manquant dans la requête")
            return jsonify({'error': 'Le champ "text" est manquant dans la requête'}), 400

        text_data = data["text"]

        # Faire la prédiction
        predictions = preprocess_and_predict([text_data])
        logger.info(f"Prédictions : {predictions}")

        # Extraire la première (et seule) valeur en int
        prediction_value = int(predictions[0][0])

        logger.info(f"prediction_value : {prediction_value}")
        # Retourner un dict avec un int simple sous "prediction"
        return jsonify({'prediction': prediction_value})

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400
    
@app.route('/report-wrong-prediction', methods=['POST'])
def report_wrong_prediction():
    data = request.get_json()
    tweet = data.get("text")
    prediction = data.get("prediction")
    logger.warning(f"TWEET MAL PREDIT : '{tweet}' --> Prediction: {prediction}")
    return jsonify({"message": "Signalement enregistré"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)