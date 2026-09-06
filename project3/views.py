import os
import json
import pickle
import random

import numpy as np
import joblib
from django.shortcuts import render
from datasets import load_dataset


# Task 1: load pre-trained classifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ag_news_model.pkl")

with open(MODEL_PATH, "rb") as f:
    _model_bundle = pickle.load(f)

_pipeline = _model_bundle["pipeline"]
_accuracy = _model_bundle["accuracy"]
_train_samples = _model_bundle["train_samples"]
_test_samples = _model_bundle["test_samples"]


def index(request):
    context = {
        "accuracy": round(_accuracy * 100, 2),
        "train_samples": _train_samples,
        "test_samples": _test_samples,
    }
    return render(request, "project3/index.html", context)


# Task 2: simulated expert
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
STRONG_CLASSES = {1, 2}


def simulated_expert(true_label, rng=random):
    if true_label in STRONG_CLASSES:
        if rng.random() < 0.9:
            return true_label
    else:
        if rng.random() < 0.3:
            return true_label

    other_labels = [l for l in LABEL_NAMES if l != true_label]
    return rng.choice(other_labels)


# Run once at startup
_dataset = load_dataset("fancyzhx/ag_news")
_test_labels = _dataset["test"]["label"]
_test_texts = _dataset["test"]["text"]

_correct = 0
_per_class_correct = {label: 0 for label in LABEL_NAMES}
_per_class_total = {label: 0 for label in LABEL_NAMES}

for _true_label in _test_labels:
    _guess = simulated_expert(_true_label)
    _per_class_total[_true_label] += 1
    if _guess == _true_label:
        _correct += 1
        _per_class_correct[_true_label] += 1

_expert_accuracy = _correct / len(_test_labels)
_expert_per_class = {
    LABEL_NAMES[label]: round(_per_class_correct[label] / _per_class_total[label] * 100, 2)
    for label in LABEL_NAMES
}


def simulated_expert_view(request):
    context = {
        "expert_accuracy": round(_expert_accuracy * 100, 2),
        "expert_per_class": _expert_per_class,
    }
    return render(request, "project3/simulated.html", context)


# Task 3: learning-to-defer
REJECTOR_PATH = os.path.join(os.path.dirname(__file__), "rejector_model.joblib")
THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "defer_thresholds.pkl")

_rejector = joblib.load(REJECTOR_PATH)
with open(THRESHOLDS_PATH, "rb") as f:
    _per_class_thresh = pickle.load(f)

# Run the full L2D system once at startup and cache the results, same
# pattern as the Task 2 expert simulation above -- avoids recomputing
# on every page view.
_test_proba = _pipeline.predict_proba(_test_texts)
_clf_pred_test = _test_proba.argmax(axis=1)
_sorted_proba = np.sort(_test_proba, axis=1)
_test_confidence = _sorted_proba[:, -1]
_test_margin = _sorted_proba[:, -1] - _sorted_proba[:, -2]

_rng_defer = random.Random(43)  # distinct seed from Task 2's expert simulation
_expert_pred_test = np.array([simulated_expert(l, _rng_defer) for l in _test_labels])

_onehot_test = np.eye(4)[_clf_pred_test]
_X_defer_test = np.column_stack([_test_confidence, _test_margin, _onehot_test])
_test_proba_defer = _rejector.predict_proba(_X_defer_test)[:, 1]
_thresh_per_example = np.array([_per_class_thresh[c] for c in _clf_pred_test])
_defer_decision = (_test_proba_defer >= _thresh_per_example).astype(int)

_final_pred = np.where(_defer_decision == 1, _expert_pred_test, _clf_pred_test)
_y_test_arr = np.array(_test_labels)

_l2d_system_accuracy = (_final_pred == _y_test_arr).mean()
_l2d_deferral_rate = _defer_decision.mean()
_l2d_oracle_upper_bound = (
    (_clf_pred_test == _y_test_arr) | (_expert_pred_test == _y_test_arr)
).mean()

_l2d_per_class = {}
for _label, _name in LABEL_NAMES.items():
    _mask = _y_test_arr == _label
    _l2d_per_class[_name] = {
        "defer_rate": round(_defer_decision[_mask].mean() * 100, 2),
        "system_acc": round((_final_pred[_mask] == _y_test_arr[_mask]).mean() * 100, 2),
    }


def learning_to_defer_view(request):
    context = {
        "classifier_only_accuracy": round(_accuracy * 100, 2),
        "expert_only_accuracy": round(_expert_accuracy * 100, 2),
        "system_accuracy": round(_l2d_system_accuracy * 100, 2),
        "oracle_upper_bound": round(_l2d_oracle_upper_bound * 100, 2),
        "deferral_rate": round(_l2d_deferral_rate * 100, 2),
        "per_class": _l2d_per_class,
    }
    return render(request, "project3/learning_to_defer.html", context)


# Task 4: active learning for expert competence discovery
AL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "al_results.json")

with open(AL_RESULTS_PATH) as f:
    _al_results = json.load(f)


def active_learning_view(request):
    context = {
        "classifier_only_accuracy": _al_results["classifier_only_accuracy"],
        "total_available_examples": _al_results["total_available_examples"],
        "recommended_strategy": _al_results["recommended_strategy"],
        "recommended_budget": _al_results["recommended_budget"],
        "recommended_result": _al_results["recommended_result"],
        # zipped rows for a single comparison table: one row per query budget
        "curve_rows": [
            {
                "queries": u["queries"],
                "uncertainty_acc": u["system_accuracy"],
                "random_acc": r["system_accuracy"],
                "hybrid_acc": h["system_accuracy"],
            }
            for u, r, h in zip(
                _al_results["curves"]["uncertainty"],
                _al_results["curves"]["random"],
                _al_results["curves"]["hybrid"],
            )
        ],
    }
    return render(request, "project3/active_learning.html", context)