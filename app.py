from flask import Flask, request, jsonify
import joblib
from src.preprocessing import preprocess_text

app = Flask(__name__)

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data["message"]

    clean_msg = preprocess_text(message)
    vectorized_msg = vectorizer.transform([clean_msg])

    pred = model.predict(vectorized_msg)[0]
    label = "Spam" if pred == 1 else "Not Spam"

    return jsonify({
        "message": message,
        "prediction": label
    })

if __name__ == "__main__":
    app.run(debug=True)
