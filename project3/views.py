import os
import pickle
import random

from django.shortcuts import render
from datasets import load_dataset


# ============================================================
# CLASSIFIER
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "ag_news_model.pkl"
)

with open(MODEL_PATH, "rb") as f:
    _model_bundle = pickle.load(f)

_model = _model_bundle["pipeline"]

_classifier_accuracy = _model_bundle["accuracy"]
_train_samples = _model_bundle["train_samples"]
_test_samples = _model_bundle["test_samples"]


# ============================================================
# DATASET
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
# CLASSIFIER CATEGORY ACCURACY
# ============================================================

# Make predictions once when Django starts
_classifier_predictions = _model.predict(_test_texts)

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

STRONG_CLASSES = {1, 2}


def simulated_expert(true_label):
    """
    Simulated expert:
    - Strong on Sports and Business
    - Weaker on World and Sci/Tech
    """

    if true_label in STRONG_CLASSES:
        if random.random() < 0.9:
            return true_label

    else:
        if random.random() < 0.3:
            return true_label

    other_labels = [
        label for label in LABEL_NAMES
        if label != true_label
    ]

    return random.choice(other_labels)


# ============================================================
# EXPERT EVALUATION
# ============================================================

# Keep expert results stable between server restarts
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


_expert_accuracy = _expert_correct / len(_test_labels)


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
# HOME
# ============================================================

def index(request):

    context = {
        "accuracy": round(
            _classifier_accuracy * 100,
            2
        ),
        "train_samples": _train_samples,
        "test_samples": _test_samples,
    }

    return render(
        request,
        "project3/index.html",
        context
    )


# ============================================================
# ACTIVE LEARNING COMPARISON PAGE
# ============================================================

def comparison(request):

    selected_category = request.GET.get(
        "category",
        "All"
    )

    # -------------------------------
    # Overall values
    # -------------------------------

    classifier_accuracy = round(
        _classifier_accuracy * 100,
        2
    )

    expert_accuracy = round(
        _expert_accuracy * 100,
        2
    )

    average_accuracy = round(
        (classifier_accuracy + expert_accuracy) / 2,
        2
    )


    # -------------------------------
    # Category values
    # -------------------------------

    if selected_category == "All":

        category_classifier_accuracy = classifier_accuracy
        category_expert_accuracy = expert_accuracy

    else:

        category_classifier_accuracy = (
            _classifier_per_class.get(
                selected_category,
                0
            )
        )

        category_expert_accuracy = (
            _expert_per_class.get(
                selected_category,
                0
            )
        )


    context = {

        # Page information
        "selected_category": selected_category,

        "categories": [
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        ],

        # Overall
        "classifier_accuracy": classifier_accuracy,
        "expert_accuracy": expert_accuracy,
        "average_accuracy": average_accuracy,

        # Selected category
        "category_classifier_accuracy":
            category_classifier_accuracy,

        "category_expert_accuracy":
            category_expert_accuracy,

        # Dataset
        "test_samples": _test_samples,

    }

    return render(
        request,
        "project3/comparison.html",
        context
    )


# ============================================================
# SIMULATED EXPERT PAGE
# ============================================================

def simulated_expert_view(request):

    context = {
        "expert_accuracy": round(
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