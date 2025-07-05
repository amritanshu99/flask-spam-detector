from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# 🔁 Load the saved model and vectorizer
try:
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
except Exception as e:
    raise RuntimeError(f"Error loading model/vectorizer: {e}")

# 🚀 Initialize Flask app
app = Flask(__name__)

# 🌐 Enable CORS for all origins (customize if needed)
CORS(app)

# 🔗 Root endpoint to check health
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "✅ Flask Spam API is running!"}), 200

# 📨 Predict endpoint (strict match for /predict)
@app.route('/predict', methods=['POST'])
def predict_spam():
    try:
        # 📩 Ensure JSON input
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        # 🧾 Get subject and body
        data = request.get_json()
        subject = data.get("subject", "")
        body = data.get("body", "")

        # 📦 Combine and clean
        full_text = f"{subject} {body}".strip()
        if not full_text:
            return jsonify({"error": "Email subject and body cannot both be empty"}), 400

        # 🔠 Transform using TF-IDF
        vector = vectorizer.transform([full_text])

        # 🤖 Predict
        prediction = model.predict(vector)[0]

        return jsonify({"spam": bool(prediction)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔌 Run locally (ignored by Render)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
