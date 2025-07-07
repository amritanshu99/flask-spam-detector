from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import logging
import sys
import os
import traceback

# ✅ Setup logging to stdout (important for Render logs)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('spam_api')

# 🔇 Suppress default Werkzeug logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# 📁 Model and vectorizer paths
MODEL_PATH = os.path.join(os.getcwd(), "spam_model.pkl")
VEC_PATH = os.path.join(os.getcwd(), "tfidf_vectorizer.pkl")

# 🔁 Load model and vectorizer
try:
    logger.info("📦 Loading model and vectorizer...")
    print("📦 Checking model and vectorizer paths...")
    print("Model Exists:", os.path.exists(MODEL_PATH))
    print("Vectorizer Exists:", os.path.exists(VEC_PATH))

    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VEC_PATH, "rb"))

    logger.info("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    logger.exception("❌ Failed to load model or vectorizer")
    print("❌ Exception during model/vectorizer loading")
    traceback.print_exc()
    raise RuntimeError(f"Error loading model/vectorizer: {e}")

# 🚀 Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Flask Spam API is running!"}), 200

@app.route("/ping", methods=["GET"])
def ping():
    return "", 204

@app.route("/predict", methods=["POST"])
def predict_spam():
    try:
        logger.info("📩 Received /predict request")
        print("📩 Received /predict request")

        if not request.is_json:
            logger.warning("❌ Request not in JSON format")
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()
        subject = data.get("subject", "")
        body = data.get("body", "")
        logger.debug(f"🔍 Subject: {subject}, Body: {body}")
        print(f"🔍 Subject: {subject}, Body: {body}")

        full_text = f"{subject} {body}".strip()
        if not full_text:
            logger.warning("❌ Empty subject and body")
            return jsonify({"error": "Email subject and body cannot both be empty"}), 400

        logger.info("🔠 Transforming text with TF-IDF")
        vector = vectorizer.transform([full_text])

        logger.info("🤖 Making prediction")
        prediction = model.predict(vector)[0]

        logger.info(f"✅ Prediction complete: {'SPAM' if prediction else 'NOT SPAM'}")
        return jsonify({"spam": bool(prediction)}), 200

    except Exception as e:
        logger.exception("❌ Error during prediction")
        print("❌ Exception caught in /predict route")
        traceback.print_exc()
        return jsonify({"error": "Failed to get spam prediction"}), 500

if __name__ == "__main__":
    print("🚀 Flask app starting locally or on Render")
    app.run(host="0.0.0.0", port=5000)
