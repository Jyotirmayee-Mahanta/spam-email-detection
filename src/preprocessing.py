import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=3000
)

# Transform text into TF-IDF features
X = vectorizer.fit_transform(df['message'])
y = df['label']

print("TF-IDF Shape:", X.shape)
print("Labels Shape:", y.shape)
