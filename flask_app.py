from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import logging
import sys
import os
import traceback

# ✅ Configure logging (INFO level is safe for Render/cron)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('spam_api')

# 🔇 Suppress default Werkzeug logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# 📁 Paths
MODEL_PATH = os.path.join(os.getcwd(), "spam_model.pkl")
VEC_PATH = os.path.join(os.getcwd(), "tfidf_vectorizer.pkl")

# 🔁 Load model and vectorizer with error handling
try:
    logger.info("📦 Loading model and vectorizer...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        raise FileNotFoundError("Model or vectorizer file not found.")
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VEC_PATH, "rb"))
    logger.info("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    logger.exception("❌ Failed to load model or vectorizer")
    raise RuntimeError(f"Error loading model/vectorizer: {e}")

# 🚀 Flask app setup
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Flask Spam API is running!"}), 200

@app.route("/ping", methods=["GET"])
def ping():
    logger.info("🔁 /ping received")
    return "", 204  # Used for cron-job.org pings

@app.route("/predict", methods=["POST"])
def predict_spam():
    try:
        logger.info("📩 Received /predict request")

        if not request.is_json:
            logger.warning("❌ Request not in JSON format")
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()
        subject = data.get("subject", "").strip()
        body = data.get("body", "").strip()
        full_text = f"{subject} {body}".strip()

        if not full_text:
            logger.warning("❌ Empty subject and body")
            return jsonify({"error": "Email subject and body cannot both be empty"}), 400

        logger.info("🔠 Vectorizing email content")
        vector = vectorizer.transform([full_text])

        logger.info("🤖 Predicting spam")
        prediction = model.predict(vector)[0]

        logger.info(f"✅ Prediction complete: {'SPAM' if prediction else 'NOT SPAM'}")
        return jsonify({"spam": bool(prediction)}), 200

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        traceback.print_exc(limit=1)
        return jsonify({"error": "Failed to get spam prediction"}), 500

# ❌ DO NOT INCLUDE app.run() for Render
# Render uses Gunicorn, which imports `app` object automatically
# Keep this file named `flask_app.py` and Render's start command as:
# gunicorn flask_app:app --bind 0.0.0.0:$PORT
