import os
import json
import pickle
import random

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ag_news_model.pkl")
OOF_CACHE = os.path.join(BASE_DIR, "al_oof_cache.npz")
RESULTS_PATH = os.path.join(BASE_DIR, "al_results.json")

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
STRONG_CLASSES = {1, 2}
BUDGETS = [200, 500, 1000, 2000, 5000, 10000, 20000, 40000]
RECOMMENDED_BUDGET = 10000  # queries used for the headline "chosen" result


def simulated_expert(true_label, rng):
    if true_label in STRONG_CLASSES:
        if rng.random() < 0.9:
            return true_label
    else:
        if rng.random() < 0.3:
            return true_label
    other_labels = [l for l in LABEL_NAMES if l != true_label]
    return rng.choice(other_labels)


def get_oof_predictions(X_train_text, y_train):
    if os.path.exists(OOF_CACHE):
        d = np.load(OOF_CACHE)
        return d["oof_pred"], d["oof_confidence"], d["oof_margin"]
    cv_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=50000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_proba = cross_val_predict(cv_pipeline, X_train_text, y_train, cv=skf, method="predict_proba")
    oof_pred = oof_proba.argmax(axis=1)
    sorted_proba = np.sort(oof_proba, axis=1)
    oof_confidence = sorted_proba[:, -1]
    oof_margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    np.savez(OOF_CACHE, oof_pred=oof_pred, oof_confidence=oof_confidence, oof_margin=oof_margin)
    return oof_pred, oof_confidence, oof_margin


def train_and_eval_rejector(queried_idx, oof_pred, oof_confidence, oof_margin,
                             expert_answers, y_train, clf_pred_test, test_confidence,
                             test_margin, expert_pred_test, y_test):
    n_q = len(queried_idx)
    onehot = np.eye(4)[oof_pred[queried_idx]]
    X_defer = np.column_stack([oof_confidence[queried_idx], oof_margin[queried_idx], onehot])
    expert_q = expert_answers[queried_idx]
    clf_correct_q = (oof_pred[queried_idx] == y_train[queried_idx])
    expert_correct_q = (expert_q == y_train[queried_idx])
    defer_target = ((~clf_correct_q) & expert_correct_q).astype(int)

    rejector, thresh = None, 1.01
    if 0 < defer_target.sum() < n_q:
        rng = np.random.RandomState(RANDOM_SEED)
        perm = rng.permutation(n_q)
        split = max(1, int(0.7 * n_q))
        fit_i, val_i = perm[:split], perm[split:]
        if len(val_i) > 0 and len(np.unique(defer_target[fit_i])) >= 2:
            rejector = LogisticRegression(max_iter=1000)
            rejector.fit(X_defer[fit_i], defer_target[fit_i])
            val_proba = rejector.predict_proba(X_defer[val_i])[:, 1]
            val_clf, val_exp, val_y = (
                oof_pred[queried_idx][val_i], expert_q[val_i], y_train[queried_idx][val_i]
            )
            best_t, best_acc = 1.01, accuracy_score(val_y, val_clf)
            for t in np.arange(0.05, 0.96, 0.05):
                defer_t = val_proba >= t
                pred_t = np.where(defer_t, val_exp, val_clf)
                acc_t = accuracy_score(val_y, pred_t)
                if acc_t > best_acc:
                    best_acc, best_t = acc_t, t
            thresh = best_t
        else:
            rejector = None

    if rejector is None:
        defer_decision = np.zeros(len(y_test), dtype=int)
    else:
        onehot_test = np.eye(4)[clf_pred_test]
        X_defer_test = np.column_stack([test_confidence, test_margin, onehot_test])
        proba_defer_test = rejector.predict_proba(X_defer_test)[:, 1]
        defer_decision = (proba_defer_test >= thresh).astype(int)

    final_pred = np.where(defer_decision == 1, expert_pred_test, clf_pred_test)
    return float(accuracy_score(y_test, final_pred)), float(defer_decision.mean())


def hybrid_order(oof_confidence, n):
    unc = np.argsort(oof_confidence)
    rng = np.random.RandomState(RANDOM_SEED + 2)
    rand = rng.permutation(n)
    seen = set()
    order = []
    i = j = 0
    while len(order) < n:
        if i < len(unc):
            while unc[i] in seen and i < len(unc) - 1:
                i += 1
            if unc[i] not in seen:
                order.append(unc[i]); seen.add(unc[i]); i += 1
        if j < len(rand):
            while rand[j] in seen and j < len(rand) - 1:
                j += 1
            if rand[j] not in seen:
                order.append(rand[j]); seen.add(rand[j]); j += 1
    return np.array(order)


def main():
    dataset = load_dataset("fancyzhx/ag_news")
    X_train_text = dataset["train"]["text"]
    y_train = np.array(dataset["train"]["label"])
    X_test_text = dataset["test"]["text"]
    y_test = np.array(dataset["test"]["label"])
    n = len(y_train)

    print("Computing (or loading cached) out-of-fold classifier predictions...")
    oof_pred, oof_confidence, oof_margin = get_oof_predictions(X_train_text, y_train)

    rng_expert = random.Random(RANDOM_SEED)
    expert_answers = np.array([simulated_expert(l, rng_expert) for l in y_train])

    with open(MODEL_PATH, "rb") as f:
        final_pipeline = pickle.load(f)["pipeline"]

    test_proba = final_pipeline.predict_proba(X_test_text)
    clf_pred_test = test_proba.argmax(axis=1)
    sorted_test_proba = np.sort(test_proba, axis=1)
    test_confidence = sorted_test_proba[:, -1]
    test_margin = sorted_test_proba[:, -1] - sorted_test_proba[:, -2]
    rng_test_expert = random.Random(RANDOM_SEED + 1)
    expert_pred_test = np.array([simulated_expert(l, rng_test_expert) for l in y_test])
    classifier_only_acc = float(accuracy_score(y_test, clf_pred_test))

    strategies = {
        "uncertainty": np.argsort(oof_confidence),
        "random": np.random.RandomState(RANDOM_SEED).permutation(n),
        "hybrid": hybrid_order(oof_confidence, n),
    }

    curves = {}
    for name, order in strategies.items():
        print(f"\n--- {name} ---")
        curve = []
        for b in BUDGETS:
            acc, defer_rate = train_and_eval_rejector(
                order[:b], oof_pred, oof_confidence, oof_margin, expert_answers, y_train,
                clf_pred_test, test_confidence, test_margin, expert_pred_test, y_test,
            )
            curve.append({"queries": b, "system_accuracy": round(acc * 100, 2), "deferral_rate": round(defer_rate * 100, 2)})
            print(f"  queries={b:6d}  acc={acc:.4f}  defer_rate={defer_rate:.4f}")
        curves[name] = curve

    recommended = next(c for c in curves["hybrid"] if c["queries"] == RECOMMENDED_BUDGET)

    results = {
        "classifier_only_accuracy": round(classifier_only_acc * 100, 2),
        "total_available_examples": n,
        "recommended_strategy": "hybrid",
        "recommended_budget": RECOMMENDED_BUDGET,
        "recommended_result": recommended,
        "curves": curves,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()