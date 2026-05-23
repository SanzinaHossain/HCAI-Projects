from django.shortcuts import render
from django.conf import settings
from palmerpenguins import load_penguins

import os
import numpy as np
import pandas as pd
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline





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

#.................................................Task 2 - Regularization..................................

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

#.................................................Task 3 - Logistic Regression..................................

def logistic_view(request):
    lambda_value = float(request.GET.get("lambda", 0.0))

    X, y, le_species = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    c_values = [0.001, 0.01, 0.1, 1, 10, 100]

    results = []
    best_score = -999
    best_model = None

    for c in c_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c,
                max_iter=1000,
            )
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logistic_model = model.named_steps["logisticregression"]
        complexity = abs(logistic_model.coef_).sum()

        score = accuracy - lambda_value * complexity

        results.append({
            "c": c,
            "accuracy": round(accuracy * 100, 2),
            "complexity": round(complexity, 4),
            "score": round(score, 4),
        })

        if score > best_score:
            best_score = score
            best_model = model
            best_accuracy = accuracy
            best_complexity = complexity
            best_c = c

    context = {
        "lambda_value": lambda_value,
        "accuracy": round(best_accuracy * 100, 2),
        "complexity": round(best_complexity, 4),
        "best_c": best_c,
        "best_score": round(best_score, 4),
        "results": results,
    }

    return render(request, "project2/logistic.html", context)

# .................................................Task 4 - Counterfactual..................................

def get_best_tree_model(X_train, X_test, y_train, y_test, lambda_value):
    max_leaf_options = [2, 3, 4, 5, 6, 8, 10, 15, 20]

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

        if score > best_score:
            best_score = score
            best_model = model

    return best_model


def get_best_logistic_model(X_train, X_test, y_train, y_test, lambda_value):
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]

    best_score = -999
    best_model = None

    for c in c_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c,
                max_iter=1000
            )
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logistic_model = model.named_steps["logisticregression"]
        complexity = np.abs(logistic_model.coef_).sum()

        score = accuracy - lambda_value * complexity

        if score > best_score:
            best_score = score
            best_model = model

    return best_model


def generate_counterfactuals(
    model,
    original_x,
    target_label,
    X,
    le_species,
    n_samples=300,
    k=5
):
    numeric_features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "year"
    ]

    categorical_features = [
        "island",
        "sex"
    ]

    mad = (
        X[numeric_features] - X[numeric_features].mean()
    ).abs().mean()

    mad = mad.replace(0, 1)

    candidates = []

    for _ in range(n_samples):
        new_x = original_x.copy()

        for feature in numeric_features:
            std = X[feature].std()
            noise = np.random.normal(0, 0.15 * std)

            new_x[feature] = new_x[feature] + noise
            new_x[feature] = np.clip(
                new_x[feature],
                X[feature].min(),
                X[feature].max()
            )

        for feature in categorical_features:
            if np.random.rand() < 0.3:
                possible_values = X[feature].unique()
                new_x[feature] = np.random.choice(possible_values)

        new_df = pd.DataFrame([new_x], columns=X.columns)

        prediction = model.predict(new_df)[0]

        if prediction == target_label:
            distance = 0

            for feature in numeric_features:
                distance += abs(
                    new_x[feature] - original_x[feature]
                ) / mad[feature]

            for feature in categorical_features:
                if new_x[feature] != original_x[feature]:
                    distance += 1

            candidates.append((distance, new_x.copy()))

    candidates = sorted(candidates, key=lambda x: x[0])
    best_candidates = candidates[:k]

    rows = []

    for distance, candidate in best_candidates:
        row = candidate.to_dict()

        row["distance"] = round(distance, 4)
        row["predicted_species"] = le_species.inverse_transform(
            [target_label]
        )[0]

        rows.append(row)

    return rows


def counterfactual_view(request):
    lambda_value = float(request.GET.get("lambda", 0.0))
    model_type = request.GET.get("model_type", "tree")
    example_id = int(request.GET.get("example_id", 0))
    target_species_name = request.GET.get("target_species", "Gentoo")

    X, y, le_species = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if model_type == "tree":
        model = get_best_tree_model(
            X_train,
            X_test,
            y_train,
            y_test,
            lambda_value
        )
    else:
        model = get_best_logistic_model(
            X_train,
            X_test,
            y_train,
            y_test,
            lambda_value
        )

    original_x = X.iloc[example_id]

    original_prediction = model.predict(
        pd.DataFrame([original_x], columns=X.columns)
    )[0]

    target_label = le_species.transform([target_species_name])[0]

    counterfactuals = []

    if request.GET.get("generate") == "1":
        np.random.seed(42)

        counterfactuals = generate_counterfactuals(
            model=model,
            original_x=original_x,
            target_label=target_label,
            X=X,
            le_species=le_species,
            n_samples=300,
            k=5
        )

    original_row = original_x.to_dict()
    original_row["model_prediction"] = le_species.inverse_transform(
        [original_prediction]
    )[0]

    original_table = pd.DataFrame([original_row]).to_html(
        classes="data-table",
        index=False
    )

    counterfactual_table = pd.DataFrame(counterfactuals)

    if not counterfactual_table.empty:
        table_html = counterfactual_table.to_html(
            classes="data-table",
            index=False
        )
    else:
        table_html = None

    example_options = []

    for i in range(len(X)):
        row = X.iloc[i]
        real_species = le_species.inverse_transform([y.iloc[i]])[0]

        label = (
            f"{real_species} | "
            f"Bill: {row['bill_length_mm']} mm | "
            f"Flipper: {row['flipper_length_mm']} mm | "
            f"Mass: {row['body_mass_g']} g"
        )

        example_options.append({
            "id": i,
            "label": label,
            "selected": i == example_id
        })

    species_options = []

    for species in le_species.classes_:
        species_options.append({
            "name": species,
            "selected": species == target_species_name
        })

    model_options = [
        {
            "value": "tree",
            "label": "Decision Tree",
            "selected": model_type == "tree"
        },
        {
            "value": "logistic",
            "label": "Logistic Regression",
            "selected": model_type == "logistic"
        }
    ]

    context = {
        "lambda_value": lambda_value,
        "model_options": model_options,
        "species_options": species_options,
        "example_options": example_options,
        "original_table": original_table,
        "counterfactual_table": table_html,
    }

    return render(
        request,
        "project2/counterfactual.html",
        context
    )


def compute_pdp(model, X, feature, grid_values):
    pdp_values = []

    for value in grid_values:
        X_temp = X.copy()
        X_temp[feature] = value
        probs = model.predict_proba(X_temp)
        pdp_values.append(probs.mean(axis=0))

    return np.array(pdp_values)


def compute_ale(model, X, feature, bins=10):
    values = X[feature].values
    quantiles = np.quantile(values, np.linspace(0, 1, bins + 1))
    quantiles = np.unique(quantiles)

    ale_effects = []
    grid_centers = []

    for i in range(len(quantiles) - 1):
        lower = quantiles[i]
        upper = quantiles[i + 1]

        mask = (X[feature] >= lower) & (X[feature] <= upper)
        X_bin = X[mask]

        if len(X_bin) == 0:
            continue

        X_low = X_bin.copy()
        X_high = X_bin.copy()

        X_low[feature] = lower
        X_high[feature] = upper

        probs_low = model.predict_proba(X_low)
        probs_high = model.predict_proba(X_high)

        diff = probs_high - probs_low
        ale_effects.append(diff.mean(axis=0))
        grid_centers.append((lower + upper) / 2)

    ale_effects = np.array(ale_effects)
    accumulated = np.cumsum(ale_effects, axis=0)

    accumulated = accumulated - accumulated.mean(axis=0)

    return np.array(grid_centers), accumulated


def feature_effects_view(request):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

    lambda_value = float(request.GET.get("lambda", 0.0))
    model_type = request.GET.get("model_type", "tree")
    selected_feature = request.GET.get("feature", "bill_length_mm")

    numerical_features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    X, y, le_species = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if model_type == "tree":
        model = get_best_tree_model(
            X_train,
            X_test,
            y_train,
            y_test,
            lambda_value
        )
    else:
        model = get_best_logistic_model(
            X_train,
            X_test,
            y_train,
            y_test,
            lambda_value
        )

    grid_values = np.linspace(
        X[selected_feature].min(),
        X[selected_feature].max(),
        30
    )

    pdp_values = compute_pdp(
        model,
        X,
        selected_feature,
        grid_values
    )

    ale_x, ale_values = compute_ale(
        model,
        X,
        selected_feature,
        bins=10
    )

    pdp_path = os.path.join(settings.MEDIA_ROOT, "pdp_plot.png")
    ale_path = os.path.join(settings.MEDIA_ROOT, "ale_plot.png")

    plt.figure(figsize=(8, 5))

    for i, species in enumerate(le_species.classes_):
        plt.plot(
            grid_values,
            pdp_values[:, i],
            label=species
        )

    plt.xlabel(selected_feature)
    plt.ylabel("Average predicted probability")
    plt.title("Partial Dependence Plot")
    plt.legend()
    plt.savefig(pdp_path, bbox_inches="tight", dpi=100)
    plt.close()

    plt.figure(figsize=(8, 5))

    for i, species in enumerate(le_species.classes_):
        plt.plot(
            ale_x,
            ale_values[:, i],
            label=species
        )

    plt.xlabel(selected_feature)
    plt.ylabel("Centered accumulated effect")
    plt.title("Accumulated Local Effects Plot")
    plt.legend()
    plt.savefig(ale_path, bbox_inches="tight", dpi=100)
    plt.close()

    model_options_html = ""
    for value, label in [
        ("tree", "Decision Tree"),
        ("logistic", "Logistic Regression")
    ]:
        selected = "selected" if value == model_type else ""
        model_options_html += (
            f'<option value="{value}" {selected}>{label}</option>'
        )

    feature_options_html = ""
    for feature in numerical_features:
        selected = "selected" if feature == selected_feature else ""
        feature_options_html += (
            f'<option value="{feature}" {selected}>{feature}</option>'
        )

    context = {
        "lambda_value": lambda_value,
        "model_options_html": model_options_html,
        "feature_options_html": feature_options_html,
        "selected_feature": selected_feature,
        "pdp_plot": settings.MEDIA_URL + "pdp_plot.png?v=" + str(lambda_value) + selected_feature + model_type,
        "ale_plot": settings.MEDIA_URL + "ale_plot.png?v=" + str(lambda_value) + selected_feature + model_type,
    }

    return render(
        request,
        "project2/feature_effects.html",
        context
    )