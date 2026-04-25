import pandas as pd

from django.shortcuts import render
from .forms import CSVUploadForm


def index(request):
    form = CSVUploadForm()
    data_preview = None
    columns = None
    rows = None
    error = None

    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)

        if form.is_valid():
            csv_file = request.FILES["file"]

            try:
                df = pd.read_csv(csv_file)

                rows = df.shape[0]
                columns = list(df.columns)

                data_preview = df.head(10).to_html(
                    classes="dataset-table",
                    index=False
                )

            except Exception as e:
                error = f"Error reading CSV file: {e}"

    context = {
        "form": form,
        "data_preview": data_preview,
        "columns": columns,
        "rows": rows,
        "error": error,
    }

    return render(request, "project1/index.html", context)