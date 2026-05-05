import csv
import io
import os
import numpy as np
from matplotlib import pyplot as plt


from django.conf import settings
from django.shortcuts import render
from .forms import CSVUploadForm
from django.http import HttpResponse, JsonResponse


# For model training
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ValidationError
import json
import logging
from .ml_models import ModelTrainer
logger = logging.getLogger(__name__)

def index(request):
    form = CSVUploadForm()
    data_preview = None
    rows = None
    columns = None
    error = None

    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)

        if form.is_valid():
            csv_file = request.FILES["file"]

            try:
                df = pd.read_csv(csv_file)

                rows = df.shape[0]
                columns = list(df.columns)

                # save uploaded dataset
                upload_dir = os.path.join(settings.MEDIA_ROOT, "datasets")
                os.makedirs(upload_dir, exist_ok=True)

                file_path = os.path.join(upload_dir, csv_file.name)
                df.to_csv(file_path, index=False)

                # store path in session
                request.session["dataset_path"] = file_path

                data_preview = df.head(5).to_html(
                      classes="dataset-table",
                      index=False
                )

            except Exception as e:
                error = f"Error reading CSV file: {e}"

    return render(request, "project1/index.html", {
        "form": form,
        "data_preview": data_preview,
        "rows": rows,
        "columns": columns,
        "error": error,
    })


def train(request):
    dataset_path = request.session.get("dataset_path")

    if not dataset_path:
        return redirect("project1:index")

    df = pd.read_csv(dataset_path)

    columns = list(df.columns)
    rows = df.shape[0]

    data_preview = df.head(5).to_html(
        classes="dataset-table",
        index=False
    )

    return render(request, "project1/train.html", {
        "columns": columns,
        "rows": rows,
        "data_preview": data_preview,
    })