from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load AG News Dataset
# -----------------------------
dataset = load_dataset("ag_news")

train_text = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

test_text = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

print("Training samples :", len(train_text))
print("Testing samples  :", len(test_text))

# -----------------------------
# Build Pipeline
# -----------------------------
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=50000
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

# -----------------------------
# Train
# -----------------------------
print("\nTraining model...")
pipeline.fit(train_text, train_labels)

# -----------------------------
# Predict
# -----------------------------
predictions = pipeline.predict(test_text)

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(test_labels, predictions)

print("\nAccuracy")
print("--------------------")
print(f"{accuracy:.4f}")

print("\nClassification Report")
print("--------------------")
print(classification_report(test_labels, predictions))

print("\nConfusion Matrix")
print("--------------------")
print(confusion_matrix(test_labels, predictions))