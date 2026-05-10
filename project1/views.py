import csv
import io
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from django.conf import settings
from django.shortcuts import render, redirect
from .forms import CSVUploadForm
from django.http import HttpResponse, JsonResponse

# For model training
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

    heatmap_url = None

    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)

        if form.is_valid():
            csv_file = request.FILES["file"]

            try:
                df = pd.read_csv(csv_file)

                rows = df.shape[0]
                columns = list(df.columns)

                upload_dir = os.path.join(settings.MEDIA_ROOT, "datasets")
                os.makedirs(upload_dir, exist_ok=True)

                file_path = os.path.join(upload_dir, csv_file.name)
                df.to_csv(file_path, index=False)

                request.session["dataset_path"] = file_path

                # random 5 rows preview
                data_preview = df.sample(min(5, len(df))).to_html(
                    classes="dataset-table",
                    index=False
                )

                # ---------------------------------
                # Correlation Heatmap
                # ---------------------------------
                numeric_df = df.select_dtypes(include=np.number)

                if len(numeric_df.columns) > 1:

                    plot_dir = os.path.join(settings.MEDIA_ROOT, "plots")
                    os.makedirs(plot_dir, exist_ok=True)

                    heatmap_filename = "correlation_heatmap.png"
                    heatmap_path = os.path.join(
                        plot_dir,
                        heatmap_filename
                    )

                    plt.figure(figsize=(10, 3))

                    correlation_matrix = numeric_df.corr()

                    sns.heatmap(
                        correlation_matrix,
                        annot=True,
                        cmap="coolwarm",
                        fmt=".2f"
                    )

                    plt.title("Feature Correlation Heatmap")
                    plt.tight_layout()

                    plt.savefig(heatmap_path)
                    plt.close()

                    heatmap_url = (
                        settings.MEDIA_URL +
                        "plots/" +
                        heatmap_filename
                    )

            except Exception as e:
                error = f"Error reading CSV file: {e}"

    return render(request, "project1/index.html", {
        "form": form,
        "data_preview": data_preview,
        "rows": rows,
        "columns": columns,
        "error": error,
        "heatmap_url": heatmap_url,
    })

def train(request):
    dataset_path = request.session.get("dataset_path")
    if not dataset_path:
        return redirect("project1:index")

    trainer = ModelTrainer()
    trainer.load_data(dataset_path)

    if request.method == "POST":
        try:
            target_column    = request.POST.get('target_column')
            model_name       = request.POST.get('model_name')
            split_percentage = int(request.POST.get('split_percentage', 80))

            # ── Collect all hp_ fields from the form ──────────────────────
            hyperparams = {
                key[3:]: value          # strip the "hp_" prefix
                for key, value in request.POST.items()
                if key.startswith('hp_')
            }
            

            trainer.prepare_data(target_column)
            result = trainer.train_model(model_name, split_percentage, hyperparams)
            request.session['training_result'] = result

            df = pd.read_csv(dataset_path)
            return render(request, "project1/mtrain.html", {
                'training_result': result,
                'columns':      list(df.columns),
                'rows':         df.shape[0],
                'data_preview': df.head(5).to_html(classes="dataset-table", index=False),
            })

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            df = pd.read_csv(dataset_path)
            return render(request, "project1/mtrain.html", {
                'error':        str(e),
                'columns':      list(df.columns),
                'rows':         df.shape[0],
                'data_preview': df.head(5).to_html(classes="dataset-table", index=False),
            })

    # GET request
    df = pd.read_csv(dataset_path)
    return render(request, "project1/mtrain.html", {
        "columns":         list(df.columns),
        "rows":            df.shape[0],
        "data_preview":    df.head(5).to_html(classes="dataset-table", index=False),
        "training_result": request.session.get('training_result'),
    })