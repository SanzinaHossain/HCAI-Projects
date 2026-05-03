import pandas as pd
import os

from django.shortcuts import render, redirect
from django.conf import settings
from .forms import CSVUploadForm


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