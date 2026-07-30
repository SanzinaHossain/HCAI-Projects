from django.shortcuts import render
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def index(request):

    dataset = load_dataset("ag_news")

    train_subset = dataset["train"].shuffle(seed=42).select(range(5000))
    test_subset = dataset["test"].shuffle(seed=42).select(range(1000))

    X_train = train_subset["text"]
    y_train = train_subset["label"]

    X_test = test_subset["text"]
    y_test = test_subset["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    context = {
        "accuracy": round(accuracy * 100, 2),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    return render(request, "project3/index.html", context)