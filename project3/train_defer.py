import os
import pickle
import random

import numpy as np
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ag_news_model.pkl")
REJECTOR_PATH = os.path.join(BASE_DIR, "rejector_model.joblib")
THRESHOLDS_PATH = os.path.join(BASE_DIR, "defer_thresholds.pkl")

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
STRONG_CLASSES = {1, 2}


def simulated_expert(true_label, rng):
    if true_label in STRONG_CLASSES:
        if rng.random() < 0.9:
            return true_label
    else:
        if rng.random() < 0.3:
            return true_label
    other_labels = [l for l in LABEL_NAMES if l != true_label]
    return rng.choice(other_labels)


def main():
    dataset = load_dataset("fancyzhx/ag_news")
    X_train_text = dataset["train"]["text"]
    y_train = np.array(dataset["train"]["label"])
    print(f"Train: {len(X_train_text)}")

    # Out-of-fold classifier predictions (5-fold CV) so the rejector's
    # training labels aren't derived from a classifier that has already
    # memorized these exact examples.
    cv_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=50000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    print("Generating out-of-fold predictions (5-fold CV)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_proba = cross_val_predict(cv_pipeline, X_train_text, y_train, cv=skf, method="predict_proba")
    oof_pred = oof_proba.argmax(axis=1)
    sorted_proba = np.sort(oof_proba, axis=1)
    oof_confidence = sorted_proba[:, -1]
    oof_margin = sorted_proba[:, -1] - sorted_proba[:, -2]

    rng_train = random.Random(RANDOM_SEED)
    expert_train = np.array([simulated_expert(l, rng_train) for l in y_train])
    clf_correct_train = (oof_pred == y_train)
    expert_correct_train = (expert_train == y_train)
    defer_target_train = ((~clf_correct_train) & expert_correct_train).astype(int)

    onehot_train = np.eye(4)[oof_pred]
    X_defer_all = np.column_stack([oof_confidence, oof_margin, onehot_train])

    # Split off a validation slice purely to tune per-class defer thresholds.
    rng_split = np.random.RandomState(RANDOM_SEED)
    n = len(y_train)
    perm = rng_split.permutation(n)
    split = int(0.7 * n)
    fit_idx, val_idx = perm[:split], perm[split:]

    rejector = LogisticRegression(max_iter=1000)
    rejector.fit(X_defer_all[fit_idx], defer_target_train[fit_idx])

    val_proba_defer = rejector.predict_proba(X_defer_all[val_idx])[:, 1]
    val_clf_pred = oof_pred[val_idx]
    val_expert_pred = expert_train[val_idx]
    val_y = y_train[val_idx]

    candidate_t = np.arange(0.01, 0.96, 0.01)
    per_class_thresh = {}
    for c in range(4):
        mask = val_clf_pred == c
        if mask.sum() == 0:
            per_class_thresh[c] = 1.01
            continue
        best_t, best_acc = 1.01, accuracy_score(val_y[mask], val_clf_pred[mask])
        for t in candidate_t:
            defer_t = val_proba_defer[mask] >= t
            pred_t = np.where(defer_t, val_expert_pred[mask], val_clf_pred[mask])
            acc_t = accuracy_score(val_y[mask], pred_t)
            if acc_t > best_acc:
                best_acc, best_t = acc_t, t
        per_class_thresh[c] = best_t
        print(f"  {LABEL_NAMES[c]:10s} threshold={best_t:.2f}  val acc -> {best_acc:.4f}")

    joblib.dump(rejector, REJECTOR_PATH)
    with open(THRESHOLDS_PATH, "wb") as f:
        pickle.dump(per_class_thresh, f)
    print(f"\nSaved {REJECTOR_PATH}")
    print(f"Saved {THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()