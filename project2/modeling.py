from __future__ import annotations

import base64
import io
import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
CATEGORICAL_FEATURES = ["island", "sex", "year"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]

FIELD_LABELS = {
    "bill_length_mm": "Bill length (mm)",
    "bill_depth_mm": "Bill depth (mm)",
    "flipper_length_mm": "Flipper length (mm)",
    "body_mass_g": "Body mass (g)",
    "island": "Island",
    "sex": "Sex",
    "year": "Year",
}


def _to_jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if pd.isna(v):
        return None
    return v


@dataclass
class Candidate:
    name: str
    model_type: str
    regularization_parameter: float | int | str
    pipeline: Pipeline
    accuracy: float
    complexity: int


class ExplainabilityProject:
    """All model and explanation logic for HCAI Project 2."""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.df = self._clean(self.df)
        self.X = self.df[FEATURES].copy()
        self.y = self.df["species"].copy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.30,
            random_state=42,
            stratify=self.y,
        )
        self.tree_candidates: List[Candidate] = []
        self.logistic_candidates: List[Candidate] = []
        self._plot_cache: Dict[Tuple, str] = {}
        self._effect_cache: Dict[Tuple, dict] = {}
        self._lock = threading.Lock()
        self._fit_all_models()
        self.numeric_stats = self._numeric_stats()

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in NUMERIC_FEATURES:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(out[col].median())
        out["sex"] = out["sex"].fillna("Unknown").astype(str).str.title()
        out["island"] = out["island"].fillna("Unknown").astype(str)
        # Treat year as categorical: the brief explicitly refers to four numerical features.
        out["year"] = out["year"].fillna(out["year"].mode().iloc[0]).astype(int).astype(str)
        out["species"] = out["species"].astype(str)
        return out.reset_index(drop=True)

    @staticmethod
    def _preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )

    def _fit_pipeline(self, estimator) -> Pipeline:
        pipe = Pipeline([
            ("prep", self._preprocessor()),
            ("model", estimator),
        ])
        pipe.fit(self.X_train, self.y_train)
        return pipe

    def _fit_all_models(self):
        # Task 1: an ordinary decision tree without an explicit leaf limit.
        baseline = self._fit_pipeline(DecisionTreeClassifier(random_state=42))
        baseline_acc = accuracy_score(self.y_test, baseline.predict(self.X_test))
        baseline_leaves = baseline.named_steps["model"].get_n_leaves()
        self.baseline_tree = Candidate(
            name="Baseline tree",
            model_type="tree",
            regularization_parameter="None",
            pipeline=baseline,
            accuracy=float(baseline_acc),
            complexity=int(baseline_leaves),
        )

        # Task 2: several trees with different max_leaf_nodes values.
        for leaves in [3, 4, 5, 6, 8, 10, 12, 16, 24, 32]:
            pipe = self._fit_pipeline(
                DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=42)
            )
            acc = accuracy_score(self.y_test, pipe.predict(self.X_test))
            complexity = pipe.named_steps["model"].get_n_leaves()
            self.tree_candidates.append(Candidate(
                name=f"Tree (max {leaves} leaves)",
                model_type="tree",
                regularization_parameter=leaves,
                pipeline=pipe,
                accuracy=float(acc),
                complexity=int(complexity),
            ))

        # Task 3: L1-regularized multinomial logistic regression.
        # Complexity Ω is the total number of non-zero coefficients.
        for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
            pipe = self._fit_pipeline(LogisticRegression(
                C=c,
                penalty="l1",
                solver="saga",
                max_iter=5000,
                random_state=42,
                tol=1e-4,
            ))
            acc = accuracy_score(self.y_test, pipe.predict(self.X_test))
            coef = pipe.named_steps["model"].coef_
            complexity = int(np.count_nonzero(np.abs(coef) > 1e-8))
            self.logistic_candidates.append(Candidate(
                name=f"Logistic regression (C={c:g})",
                model_type="logistic",
                regularization_parameter=c,
                pipeline=pipe,
                accuracy=float(acc),
                complexity=complexity,
            ))

    def _numeric_stats(self):
        stats = {}
        for col in NUMERIC_FEATURES:
            s = self.X_train[col].astype(float)
            med = float(s.median())
            mad = float(np.median(np.abs(s - med)))
            if not np.isfinite(mad) or mad < 1e-9:
                mad = float(s.std()) or 1.0
            stats[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "median": med,
                "mad": mad,
            }
        return stats

    def select_candidate(self, model_type: str, lam: float, rule: str = "assignment") -> Candidate:
        candidates = self.tree_candidates if model_type == "tree" else self.logistic_candidates
        if rule == "assignment":
            # Follow Task 2 literally: minimize test accuracy + λΩ(g).
            key = lambda c: c.accuracy + lam * c.complexity
        else:
            # Optional pedagogical comparison with Equation (1): loss + penalty.
            key = lambda c: (1.0 - c.accuracy) + lam * c.complexity
        return min(candidates, key=key)

    def candidate_rows(self, model_type: str, lam: float, rule: str):
        candidates = self.tree_candidates if model_type == "tree" else self.logistic_candidates
        selected = self.select_candidate(model_type, lam, rule)
        rows = []
        for c in candidates:
            score = (c.accuracy if rule == "assignment" else (1.0 - c.accuracy)) + lam * c.complexity
            rows.append({
                "name": c.name,
                "fit_parameter": _to_jsonable(c.regularization_parameter),
                "accuracy": round(c.accuracy, 4),
                "complexity": c.complexity,
                "score": round(float(score), 5),
                "selected": c is selected,
            })
        return rows

    @staticmethod
    def _fig_to_base64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=135, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def tree_image(self, candidate: Candidate) -> str:
        key = ("tree", candidate.name)
        if key in self._plot_cache:
            return self._plot_cache[key]
        pipe = candidate.pipeline
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        feature_names = [n.replace("num__", "").replace("cat__", "") for n in prep.get_feature_names_out()]
        fig, ax = plt.subplots(figsize=(17, 8.5))
        plot_tree(
            model,
            feature_names=feature_names,
            class_names=[str(c) for c in model.classes_],
            filled=True,
            rounded=True,
            fontsize=7,
            ax=ax,
        )
        ax.set_title(f"{candidate.name} — {candidate.complexity} leaves", fontsize=13)
        encoded = self._fig_to_base64(fig)
        self._plot_cache[key] = encoded
        return encoded

    def logistic_coefficients(self, candidate: Candidate):
        pipe = candidate.pipeline
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        names = [n.replace("num__", "").replace("cat__", "") for n in prep.get_feature_names_out()]
        rows = []
        for class_idx, cls in enumerate(model.classes_):
            coef = model.coef_[class_idx]
            idxs = np.argsort(np.abs(coef))[::-1][:8]
            for idx in idxs:
                rows.append({
                    "species": str(cls),
                    "feature": str(names[idx]),
                    "coefficient": round(float(coef[idx]), 4),
                })
        return rows

    def model_state(self, model_type: str, lam: float, rule: str):
        selected = self.select_candidate(model_type, lam, rule)
        state = {
            "model_type": model_type,
            "lambda": lam,
            "rule": rule,
            "selected_name": selected.name,
            "accuracy": round(selected.accuracy, 4),
            "complexity": selected.complexity,
            "complexity_label": "number of leaves" if model_type == "tree" else "non-zero coefficients",
            "fit_parameter": _to_jsonable(selected.regularization_parameter),
            "candidate_rows": self.candidate_rows(model_type, lam, rule),
        }
        if model_type == "tree":
            state["tree_image"] = self.tree_image(selected)
        else:
            state["coefficients"] = self.logistic_coefficients(selected)
        return state

    def task1_state(self):
        return {
            "name": self.baseline_tree.name,
            "accuracy": round(self.baseline_tree.accuracy, 4),
            "leaves": self.baseline_tree.complexity,
            "tree_image": self.tree_image(self.baseline_tree),
        }

    def examples(self):
        rows = []
        for idx, r in self.df.iterrows():
            rows.append({
                "index": int(idx),
                "label": f"#{idx + 1} — {r['species']} ({r['island']}, {r['sex']}, {r['year']})",
                "species": r["species"],
            })
        return rows

    def example_detail(self, idx: int):
        r = self.df.iloc[int(idx)]
        return {k: _to_jsonable(r[k]) for k in ["species"] + FEATURES}

    def counterfactuals(
        self,
        model_type: str,
        lam: float,
        rule: str,
        example_index: int,
        target_label: str,
        k: int = 5,
        seed: int = 123,
    ):
        candidate = self.select_candidate(model_type, lam, rule)
        x = self.df.iloc[int(example_index)][FEATURES].copy()
        original_pred = str(candidate.pipeline.predict(pd.DataFrame([x]))[0])
        rng = np.random.default_rng(seed + int(example_index))

        found = []
        attempts = [
            (1200, 0.45),
            (2200, 0.80),
            (3500, 1.20),
            (5000, 1.80),
        ]
        cat_values = {c: self.X_train[c].astype(str).unique().tolist() for c in CATEGORICAL_FEATURES}

        for n, scale in attempts:
            samples = []
            for _ in range(n):
                row = x.copy()
                for col in NUMERIC_FEATURES:
                    st = self.numeric_stats[col]
                    sigma = max(st["mad"] * scale, (st["max"] - st["min"]) * 0.01)
                    value = float(x[col]) + rng.normal(0.0, sigma)
                    row[col] = float(np.clip(value, st["min"], st["max"]))
                for col in CATEGORICAL_FEATURES:
                    # Categorical features are not given Gaussian noise. They either stay
                    # unchanged or switch to another valid observed category.
                    if rng.random() < min(0.15 + scale * 0.12, 0.55):
                        row[col] = rng.choice(cat_values[col])
                samples.append(row)

            sdf = pd.DataFrame(samples, columns=FEATURES)
            pred = candidate.pipeline.predict(sdf)
            mask = pred == target_label
            if np.any(mask):
                desired = sdf.loc[mask].copy()
                for _, row in desired.iterrows():
                    dist = 0.0
                    changes = []
                    for col in NUMERIC_FEATURES:
                        mad = self.numeric_stats[col]["mad"]
                        d = abs(float(row[col]) - float(x[col])) / max(mad, 1e-9)
                        dist += d
                        if abs(float(row[col]) - float(x[col])) > 1e-8:
                            changes.append({
                                "field": col,
                                "label": FIELD_LABELS[col],
                                "from": round(float(x[col]), 2),
                                "to": round(float(row[col]), 2),
                            })
                    for col in CATEGORICAL_FEATURES:
                        if str(row[col]) != str(x[col]):
                            dist += 1.0
                            changes.append({
                                "field": col,
                                "label": FIELD_LABELS[col],
                                "from": str(x[col]),
                                "to": str(row[col]),
                            })
                    found.append((float(dist), row.copy(), changes))
                break

        found.sort(key=lambda z: z[0])
        unique = []
        seen = set()
        for dist, row, changes in found:
            signature = tuple(round(float(row[c]), 2) if c in NUMERIC_FEATURES else str(row[c]) for c in FEATURES)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append({
                "distance": round(dist, 3),
                "changes": changes,
                "values": {c: _to_jsonable(row[c]) for c in FEATURES},
            })
            if len(unique) >= k:
                break

        return {
            "model_name": candidate.name,
            "original_prediction": original_pred,
            "original_true_species": str(self.df.iloc[int(example_index)]["species"]),
            "target_label": target_label,
            "original": {c: _to_jsonable(x[c]) for c in FEATURES},
            "counterfactuals": unique,
            "found": len(unique),
            "method_note": (
                "Numeric fields are perturbed locally and categorical fields are switched only to valid observed categories. "
                "Candidates predicted as the requested species are ranked by MAD-weighted L1 distance plus one unit per categorical change."
            ),
        }

    def _pdp(self, candidate: Candidate, feature: str, grid: np.ndarray):
        curves = {c: [] for c in candidate.pipeline.classes_}
        base = self.X_test.copy().reset_index(drop=True)
        for val in grid:
            modified = base.copy()
            modified[feature] = float(val)
            probs = candidate.pipeline.predict_proba(modified)
            means = probs.mean(axis=0)
            for i, cls in enumerate(candidate.pipeline.classes_):
                curves[cls].append(float(means[i]))
        return curves

    def _ale_tree_discrete(self, candidate: Candidate, feature: str, edges: np.ndarray):
        base = self.X_train.copy().reset_index(drop=True)
        classes = candidate.pipeline.classes_
        increments = np.zeros((len(edges) - 1, len(classes)), dtype=float)
        counts = np.zeros(len(edges) - 1, dtype=float)
        xvals = base[feature].to_numpy(float)
        bin_ids = np.clip(np.digitize(xvals, edges[1:-1], right=False), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            local = base.loc[mask].copy()
            low = local.copy(); low[feature] = edges[b]
            high = local.copy(); high[feature] = edges[b + 1]
            diff = candidate.pipeline.predict_proba(high) - candidate.pipeline.predict_proba(low)
            increments[b] = diff.mean(axis=0)
            counts[b] = mask.sum()
        ale = np.cumsum(increments, axis=0)
        if counts.sum() > 0:
            center = (ale * counts[:, None]).sum(axis=0) / counts.sum()
            ale = ale - center
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, {cls: ale[:, i].tolist() for i, cls in enumerate(classes)}

    def _ale_logistic_exact(self, candidate: Candidate, feature: str, edges: np.ndarray):
        """ALE using exact multinomial-logistic partial derivatives within each bin.

        For p_k = softmax(z)_k, dp_k/dx_j = p_k*(beta_kj - sum_l p_l beta_lj)/scale_j.
        The expectation of this exact derivative is approximated by the rows in each ALE bin;
        integration over the bin uses the bin width.
        """
        base = self.X_train.copy().reset_index(drop=True)
        pipe = candidate.pipeline
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        num_idx = NUMERIC_FEATURES.index(feature)
        scale = float(prep.named_transformers_["num"].scale_[num_idx])
        beta = model.coef_[:, num_idx] / scale
        classes = model.classes_

        xvals = base[feature].to_numpy(float)
        bin_ids = np.clip(np.digitize(xvals, edges[1:-1], right=False), 0, len(edges) - 2)
        increments = np.zeros((len(edges) - 1, len(classes)), dtype=float)
        counts = np.zeros(len(edges) - 1, dtype=float)

        probs_all = pipe.predict_proba(base)
        for b in range(len(edges) - 1):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            probs = probs_all[mask]
            weighted_beta = probs @ beta
            deriv = probs * (beta[None, :] - weighted_beta[:, None])
            mean_deriv = deriv.mean(axis=0)
            increments[b] = mean_deriv * (edges[b + 1] - edges[b])
            counts[b] = mask.sum()

        ale = np.cumsum(increments, axis=0)
        if counts.sum() > 0:
            center = (ale * counts[:, None]).sum(axis=0) / counts.sum()
            ale = ale - center
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, {cls: ale[:, i].tolist() for i, cls in enumerate(classes)}

    def feature_effects(self, model_type: str, lam: float, rule: str, feature: str):
        if feature not in NUMERIC_FEATURES:
            raise ValueError("Feature must be one of the four numerical measurement features.")
        key = (model_type, round(float(lam), 6), rule, feature)
        if key in self._effect_cache:
            return self._effect_cache[key]

        candidate = self.select_candidate(model_type, lam, rule)
        values = self.X_train[feature].to_numpy(float)
        lo, hi = np.quantile(values, [0.02, 0.98])
        grid = np.linspace(lo, hi, 30)
        pdp_curves = self._pdp(candidate, feature, grid)

        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if len(edges) < 4:
            edges = np.linspace(values.min(), values.max(), 10)
        if model_type == "logistic":
            ale_x, ale_curves = self._ale_logistic_exact(candidate, feature, edges)
            ale_method = "Exact logistic-regression partial derivatives, averaged within bins and integrated."
        else:
            ale_x, ale_curves = self._ale_tree_discrete(candidate, feature, edges)
            ale_method = "Finite-difference discretization at ALE bin edges because a decision tree has stepwise predictions and no useful ordinary derivative at its splits."

        pdp_img = self._effect_plot(
            grid,
            pdp_curves,
            title=f"PDP — {FIELD_LABELS[feature]}",
            xlabel=FIELD_LABELS[feature],
            ylabel="Average predicted probability",
            zero_line=False,
        )
        ale_img = self._effect_plot(
            ale_x,
            ale_curves,
            title=f"ALE — {FIELD_LABELS[feature]}",
            xlabel=FIELD_LABELS[feature],
            ylabel="Centered accumulated effect",
            zero_line=True,
        )
        result = {
            "model_name": candidate.name,
            "feature": feature,
            "feature_label": FIELD_LABELS[feature],
            "pdp_image": pdp_img,
            "ale_image": ale_img,
            "ale_method": ale_method,
            "pdp_note": "PDP is computed manually by replacing the selected feature with each grid value for every test row, predicting all three species probabilities, and averaging those probabilities.",
        }
        self._effect_cache[key] = result
        return result

    def _effect_plot(self, x, curves, title, xlabel, ylabel, zero_line=False):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for cls in SPECIES:
            if cls in curves:
                ax.plot(x, curves[cls], marker="o", markersize=3, linewidth=2, label=cls)
        if zero_line:
            ax.axhline(0, linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(title="Species")
        fig.tight_layout()
        return self._fig_to_base64(fig)
