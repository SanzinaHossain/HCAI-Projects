import os
import pickle
import random

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
    }


    return render(
        request,
        "project3/index.html",
        context
    )


# ============================================================
# OPTIONAL SIMULATED EXPERT PAGE
# ============================================================

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