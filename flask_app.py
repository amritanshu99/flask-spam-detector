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
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "✅ Flask Spam API is running!"}), 200

# 📨 Predict endpoint (strictly matches '/predict' without newline or extra chars)
@app.route('/predict', methods=['POST'])
def predict_spam():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()
        subject = data.get("subject", "")
        body = data.get("body", "")

        # Combine subject and body
        full_text = f"{subject} {body}".strip()
        if not full_text:
            return jsonify({"error": "Email subject and body cannot both be empty"}), 400

        # Convert to vector and predict
        vector = vectorizer.transform([full_text])
        prediction = model.predict(vector)[0]
        return jsonify({"spam": bool(prediction)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🟢 Start the server
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
