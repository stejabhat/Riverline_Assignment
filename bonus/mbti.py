import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils import resample
import joblib

# --- Text Cleaner ---
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special chars & digits
    text = text.lower().strip()
    return text

# Load data
df = pd.read_csv("reddit_post.csv")
df.dropna(subset=["body", "mbti"], inplace=True)
df["body"] = df["body"].apply(clean_text)

# Drop rare MBTI types (< 200 samples)
valid_types = df["mbti"].value_counts()
valid_types = valid_types[valid_types > 200].index.tolist()
df = df[df["mbti"].isin(valid_types)]

# Balance classes (up to 1000 samples each)
balanced_df = pd.concat([
    resample(df[df["mbti"] == t], n_samples=1000, random_state=42)
    for t in valid_types
])

# Features and labels
X = balanced_df["body"]
y = balanced_df["mbti"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\n📊 New Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "mbti_classifier_balanced.joblib")
print("\n✅ Balanced model saved as 'mbti_classifier_balanced.joblib'")
