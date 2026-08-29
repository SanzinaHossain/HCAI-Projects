import os
import pickle
import random

from django.shortcuts import render
from datasets import load_dataset


# Task 1: load pre-trained classifier 

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ag_news_model.pkl")

with open(MODEL_PATH, "rb") as f:
    _model_bundle = pickle.load(f)

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


def simulated_expert(true_label):
    if true_label in STRONG_CLASSES:
        if random.random() < 0.9:
            return true_label
    else:
        if random.random() < 0.3:
            return true_label

    other_labels = [l for l in LABEL_NAMES if l != true_label]
    return random.choice(other_labels)


# Run once at startup
_dataset = load_dataset("fancyzhx/ag_news")
_test_labels = _dataset["test"]["label"]

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
