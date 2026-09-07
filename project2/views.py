from __future__ import annotations

import json
import threading
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

from .modeling import ExplainabilityProject, NUMERIC_FEATURES

_PROJECT = None
_PROJECT_LOCK = threading.Lock()


def _project() -> ExplainabilityProject:
    """Create the model service once, on the first API request."""
    global _PROJECT
    if _PROJECT is None:
        with _PROJECT_LOCK:
            if _PROJECT is None:
                csv_path = Path(__file__).resolve().parent / "data" / "penguins.csv"
                _PROJECT = ExplainabilityProject(str(csv_path))
    return _PROJECT


def _selection(request):
    model = request.GET.get("model", "tree")
    rule = request.GET.get("rule", "assignment")
    try:
        lam = float(request.GET.get("lambda", "0.01"))
    except (TypeError, ValueError):
        lam = 0.01

    if model not in {"tree", "logistic"}:
        model = "tree"
    if rule not in {"assignment", "loss"}:
        rule = "assignment"
    lam = max(0.0, min(lam, 0.05))
    return model, lam, rule


def _error_response(exc, status=400):
    return JsonResponse({"error": str(exc)}, status=status)


@require_GET
def home(request):
    return render(request, "project2/index.html")


@require_GET
def api_task1(request):
    try:
        return JsonResponse(_project().task1_state())
    except Exception as exc:
        return _error_response(exc, 500)


@require_GET
def api_model(request):
    try:
        model, lam, rule = _selection(request)
        return JsonResponse(_project().model_state(model, lam, rule))
    except Exception as exc:
        return _error_response(exc, 500)


@require_GET
def api_examples(request):
    try:
        return JsonResponse({"examples": _project().examples()})
    except Exception as exc:
        return _error_response(exc, 500)


@require_GET
def api_example(request):
    try:
        idx = int(request.GET.get("index", "0"))
        return JsonResponse(_project().example_detail(idx))
    except Exception as exc:
        return _error_response(exc)


@csrf_exempt
@require_POST
def api_counterfactual(request):
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
        result = _project().counterfactuals(
            model_type=payload.get("model", "tree"),
            lam=float(payload.get("lambda", 0.01)),
            rule=payload.get("rule", "assignment"),
            example_index=int(payload.get("example_index", 0)),
            target_label=str(payload.get("target_label", "Gentoo")),
            k=max(1, min(int(payload.get("k", 5)), 10)),
        )
        return JsonResponse(result)
    except Exception as exc:
        return _error_response(exc)


@require_GET
def api_effects(request):
    try:
        model, lam, rule = _selection(request)
        feature = request.GET.get("feature", NUMERIC_FEATURES[0])
        return JsonResponse(_project().feature_effects(model, lam, rule, feature))
    except Exception as exc:
        return _error_response(exc)
