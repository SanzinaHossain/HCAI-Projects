from django.shortcuts import render
from django.conf import settings

from palmerpenguins import load_penguins

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)


def prepare_data():
    df = load_penguins().dropna()

    le_island = LabelEncoder()
    le_sex = LabelEncoder()
    le_species = LabelEncoder()

    df["island"] = le_island.fit_transform(df["island"])
    df["sex"] = le_sex.fit_transform(df["sex"])
    df["species"] = le_species.fit_transform(df["species"])

    X = df.drop("species", axis=1)
    y = df["species"]

    return X, y, le_species


def data_view(request):
    df = load_penguins().dropna()

    shuffled_df = df.sample(frac=1, random_state=42).head(10)

    context = {
        "table": shuffled_df.to_html(classes="data-table", index=False),
        "shape": df.shape,
        "total_samples": df.shape[0],
        "total_features": df.shape[1],
    }

    return render(request, "project2/data.html", context)


def tree_view(request):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

    tree_path = os.path.join(settings.MEDIA_ROOT, "decision_tree.png")
    cm_path = os.path.join(settings.MEDIA_ROOT, "confusion_matrix.png")
    roc_path = os.path.join(settings.MEDIA_ROOT, "roc_curve.png")

    X, y, le_species = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_leaf_nodes=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    leaves = model.get_n_leaves()

    images_exist = (
        os.path.exists(tree_path)
        and os.path.exists(cm_path)
        and os.path.exists(roc_path)
    )

    if not images_exist:
        # Decision Tree image
        plt.figure(figsize=(12, 7))

        plot_tree(
            model,
            feature_names=X.columns,
            class_names=le_species.classes_,
            filled=True,
            rounded=True,
            fontsize=8
        )

        plt.savefig(tree_path, bbox_inches="tight", dpi=100)
        plt.close()

        # Confusion Matrix image
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=le_species.classes_
        )

        disp.plot(
            ax=ax,
            cmap="Purples",
            values_format="d"
        )

        plt.title("Confusion Matrix")
        plt.savefig(cm_path, bbox_inches="tight", dpi=100)
        plt.close()

        # ROC Curve image
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

        plt.figure(figsize=(7, 5))

        for i, class_name in enumerate(le_species.classes_):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                label=f"{class_name} AUC = {roc_auc:.2f}"
            )

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        plt.savefig(roc_path, bbox_inches="tight", dpi=100)
        plt.close()

    context = {
    "accuracy": round(accuracy * 100, 2),
    "leaves": leaves,
    "tree_image": settings.MEDIA_URL + "decision_tree.png?v=1",
    "confusion_matrix": settings.MEDIA_URL + "confusion_matrix.png?v=1",
    "roc_curve": settings.MEDIA_URL + "roc_curve.png?v=1",
    }
    return render(request, "project2/tree.html", context)

def regularization_view(request):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

    lambda_value = float(request.GET.get("lambda", 0.0))

    tree_path = os.path.join(settings.MEDIA_ROOT, "regularized_tree.png")

    X, y, le_species = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    max_leaf_options = [2, 3, 4, 5, 6, 8, 10, 15, 20]

    results = []
    best_score = -999
    best_model = None

    for max_leaf in max_leaf_options:
        model = DecisionTreeClassifier(
            max_leaf_nodes=max_leaf,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        leaves = model.get_n_leaves()

        score = accuracy - lambda_value * leaves

        results.append({
            "max_leaf": max_leaf,
            "accuracy": round(accuracy * 100, 2),
            "leaves": leaves,
            "score": round(score, 4),
        })

        if score > best_score:
            best_score = score
            best_model = model
            best_accuracy = accuracy
            best_leaves = leaves
            best_max_leaf = max_leaf

    plt.figure(figsize=(12, 7))

    plot_tree(
        best_model,
        feature_names=X.columns,
        class_names=le_species.classes_,
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.savefig(tree_path, bbox_inches="tight", dpi=100)
    plt.close()

    context = {
        "lambda_value": lambda_value,
        "accuracy": round(best_accuracy * 100, 2),
        "leaves": best_leaves,
        "best_score": round(best_score, 4),
        "best_max_leaf": best_max_leaf,
        "results": results,
        "tree_image": settings.MEDIA_URL + "regularized_tree.png?v=" + str(lambda_value),
    }

    return render(request, "project2/regularization.html", context)