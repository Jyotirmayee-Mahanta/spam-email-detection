import joblib

# Load trained model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

def predict_spam(message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    return "SPAM" if prediction[0] == 1 else "HAM"

# Test with sample input
if __name__ == "__main__":
    sample_email = "Congratulations! You have won a free gift card."
    result = predict_spam(sample_email)
    print("Prediction:", result)
