import os
import json
import pickle
import random

import numpy as np
import joblib
from django.shortcuts import render
from datasets import load_dataset


# ============================================================
# LOAD CLASSIFIER
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "ag_news_model.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model_bundle = pickle.load(f)

_model = model_bundle["pipeline"]

_classifier_accuracy = model_bundle["accuracy"]
_train_samples = model_bundle["train_samples"]
_test_samples = model_bundle["test_samples"]


# ============================================================
# AG NEWS DATASET
# ============================================================

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

_dataset = load_dataset("fancyzhx/ag_news")

_test_texts = _dataset["test"]["text"]
_test_labels = _dataset["test"]["label"]


# ============================================================
# CLASSIFIER PREDICTIONS
# ============================================================

_classifier_predictions = _model.predict(_test_texts)


# ============================================================
# CLASSIFIER CATEGORY ACCURACY
# ============================================================

_classifier_per_class_correct = {
    label: 0 for label in LABEL_NAMES
}

_classifier_per_class_total = {
    label: 0 for label in LABEL_NAMES
}

for true_label, prediction in zip(
    _test_labels,
    _classifier_predictions
):

    _classifier_per_class_total[true_label] += 1

    if prediction == true_label:
        _classifier_per_class_correct[true_label] += 1


_classifier_per_class = {}

for label in LABEL_NAMES:

    accuracy = (
        _classifier_per_class_correct[label]
        / _classifier_per_class_total[label]
        * 100
    )

    _classifier_per_class[LABEL_NAMES[label]] = round(
        accuracy,
        2
    )


# ============================================================
# SIMULATED EXPERT
# ============================================================

# Expert is deliberately stronger in Sports and Business.
STRONG_CLASSES = {1, 2}


def simulated_expert(true_label):

    # Strong expertise
    if true_label in STRONG_CLASSES:

        if random.random() < 0.9:
            return true_label

    # Weak expertise
    else:

        if random.random() < 0.3:
            return true_label

    # If expert is wrong, choose another category
    other_labels = [
        label
        for label in LABEL_NAMES
        if label != true_label
    ]

    return random.choice(other_labels)


# ============================================================
# EXPERT EVALUATION
# ============================================================

random.seed(42)

_expert_correct = 0

_expert_per_class_correct = {
    label: 0 for label in LABEL_NAMES
}

_expert_per_class_total = {
    label: 0 for label in LABEL_NAMES
}


for true_label in _test_labels:

    expert_prediction = simulated_expert(true_label)

    _expert_per_class_total[true_label] += 1

    if expert_prediction == true_label:

        _expert_correct += 1

        _expert_per_class_correct[true_label] += 1


_expert_accuracy = (
    _expert_correct / len(_test_labels)
)


_expert_per_class = {}

for label in LABEL_NAMES:

    accuracy = (
        _expert_per_class_correct[label]
        / _expert_per_class_total[label]
        * 100
    )

    _expert_per_class[LABEL_NAMES[label]] = round(
        accuracy,
        2
    )


# ============================================================
# MAIN PAGE
# ============================================================

def index(request):

    # --------------------------------------------------------
    # OVERALL ACCURACY
    # --------------------------------------------------------

    classifier_accuracy = round(
        _classifier_accuracy * 100,
        2
    )

    expert_accuracy = round(
        _expert_accuracy * 100,
        2
    )


    # --------------------------------------------------------
    # CATEGORY TABLE
    # --------------------------------------------------------

    category_data = []

    for category in LABEL_NAMES.values():

        category_data.append({
            "name": category,

            "classifier":
                _classifier_per_class[category],

            "expert":
                _expert_per_class[category],
        })


    # --------------------------------------------------------
    # SELECTED NEWS
    # --------------------------------------------------------

    selected_news = request.GET.get(
        "news",
        "0"
    )

    try:
        news_index = int(selected_news)

    except (ValueError, TypeError):
        news_index = 0


    if news_index < 0 or news_index >= len(_test_texts):
        news_index = 0


    news_text = _test_texts[news_index]

    true_label = _test_labels[news_index]


    # --------------------------------------------------------
    # CLASSIFIER PREDICTION
    # --------------------------------------------------------

    classifier_prediction = int(
        _classifier_predictions[news_index]
    )


    # --------------------------------------------------------
    # EXPERT PREDICTION
    # --------------------------------------------------------

    # Same prediction every time for the same article
    random.seed(42 + news_index)

    expert_prediction = simulated_expert(
        true_label
    )


    # --------------------------------------------------------
    # NEWS RESULT
    # --------------------------------------------------------

    news_result = {

        "text":
            news_text,

        "true_category":
            LABEL_NAMES[true_label],

        "classifier":
            LABEL_NAMES[classifier_prediction],

        "expert":
            LABEL_NAMES[expert_prediction],

        "classifier_correct":
            classifier_prediction == true_label,

        "expert_correct":
            expert_prediction == true_label,

        # Task 3: what the learning-to-defer system actually decided for
        # this specific article. _defer_decision and _final_pred are
        # defined further down in this file (Task 3 section), but that's
        # fine -- index() only reads them when a request comes in, by
        # which point the whole module has already finished loading.
        "deferred":
            bool(_defer_decision[news_index]),

        "final_prediction":
            LABEL_NAMES[int(_final_pred[news_index])],

        "final_correct":
            int(_final_pred[news_index]) == true_label,
    }


    # --------------------------------------------------------
    # NEWS SELECTOR
    # --------------------------------------------------------

    news_options = []

    # 500 news articles available
    for i, text in enumerate(_test_texts[:500]):

        news_options.append({
            "index": i,
            "text": text[:100] + "..."
        })


    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = {

        "classifier_accuracy":
            classifier_accuracy,

        "expert_accuracy":
            expert_accuracy,

        "category_data":
            category_data,

        "news_options":
            news_options,

        "selected_news":
            news_index,

        "news_result":
            news_result,

        "train_samples":
            _train_samples,

        "test_samples":
            _test_samples,

        # Task 3: aggregate learning-to-defer performance 
        "l2d_system_accuracy":
            round(_l2d_system_accuracy * 100, 2),

        "l2d_oracle_upper_bound":
            round(_l2d_oracle_upper_bound * 100, 2),

        "l2d_deferral_rate":
            round(_l2d_deferral_rate * 100, 2),

        "l2d_per_class":
            _l2d_per_class,
    }


    return render(
        request,
        "project3/index.html",
        context
    )




def simulated_expert_view(request):

    context = {

        "expert_accuracy":
            round(
                _expert_accuracy * 100,
                2
            ),

        "expert_per_class":
            _expert_per_class,
    }

    return render(
        request,
        "project3/simulated.html",
        context
    )
# Task 3: learning-to-defer
REJECTOR_PATH = os.path.join(os.path.dirname(__file__), "rejector_model.joblib")
THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "defer_thresholds.pkl")
 
_rejector = joblib.load(REJECTOR_PATH)
with open(THRESHOLDS_PATH, "rb") as f:
    _per_class_thresh = pickle.load(f)
 

_test_proba = _model.predict_proba(_test_texts)
_clf_pred_test = _test_proba.argmax(axis=1)
_sorted_proba = np.sort(_test_proba, axis=1)
_test_confidence = _sorted_proba[:, -1]
_test_margin = _sorted_proba[:, -1] - _sorted_proba[:, -2]
 

random.seed(43)
_expert_pred_test = np.array([simulated_expert(l) for l in _test_labels])
 
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