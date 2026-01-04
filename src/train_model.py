import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Convert labels to binary
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Vectorization
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=3000
)

X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully")
