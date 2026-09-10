from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from .forms import DatasetUploadForm, TrainingForm
from .ml_models import train_model


SESSION_FILE_KEY = "project1_dataset_path"
SESSION_NAME_KEY = "project1_dataset_name"


def _get_upload_directory() -> Path:
    """
    Return the folder used to store uploaded Project 1 datasets.
    """

    media_root = Path(
        getattr(
            settings,
            "MEDIA_ROOT",
            Path(settings.BASE_DIR) / "media",
        )
    )

    upload_directory = media_root / "project1_uploads"
    upload_directory.mkdir(parents=True, exist_ok=True)

    return upload_directory


def _has_uploaded_dataset(request) -> bool:
    """
    Return True only when the session contains a valid dataset file.

    This value controls whether Review and Train navigation links
    should be enabled.
    """

    stored_path = request.session.get(SESSION_FILE_KEY)

    if not stored_path:
        return False

    dataset_path = Path(stored_path)

    if dataset_path.exists() and dataset_path.is_file():
        return True

    # Remove invalid or stale session values.
    request.session.pop(SESSION_FILE_KEY, None)
    request.session.pop(SESSION_NAME_KEY, None)
    request.session.modified = True

    return False


def _clear_uploaded_dataset(request) -> None:
    """
    Remove the uploaded dataset file and clear its session values.
    """

    stored_path = request.session.pop(SESSION_FILE_KEY, None)
    request.session.pop(SESSION_NAME_KEY, None)
    request.session.modified = True

    if stored_path:
        try:
            Path(stored_path).unlink(missing_ok=True)
        except OSError:
            pass


def _read_dataframe(request) -> pd.DataFrame:
    """
    Read and validate the currently uploaded CSV file.
    """

    stored_path = request.session.get(SESSION_FILE_KEY)

    if not stored_path:
        raise FileNotFoundError(
            "Please upload a dataset before continuing."
        )

    dataset_path = Path(stored_path)

    if not dataset_path.exists():
        request.session.pop(SESSION_FILE_KEY, None)
        request.session.pop(SESSION_NAME_KEY, None)
        request.session.modified = True

        raise FileNotFoundError(
            "The uploaded dataset is no longer available. "
            "Please upload it again."
        )

    try:
        dataframe = pd.read_csv(dataset_path)

    except UnicodeDecodeError:
        try:
            dataframe = pd.read_csv(
                dataset_path,
                encoding="latin-1",
            )
        except Exception as exc:
            raise ValueError(
                f"The CSV file could not be read: {exc}"
            ) from exc

    except Exception as exc:
        raise ValueError(
            f"The CSV file could not be read: {exc}"
        ) from exc

    if dataframe.empty:
        raise ValueError("The CSV file is empty.")

    if dataframe.shape[1] < 2:
        raise ValueError(
            "The dataset must contain at least two columns."
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


def _figure_to_data_uri(fig) -> str:
    """
    Convert a Matplotlib figure into a base64 image URI.
    """

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=140,
    )

    plt.close(fig)

    buffer.seek(0)

    encoded_image = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded_image}"


def _build_heatmap(
    dataframe: pd.DataFrame,
) -> str | None:
    """
    Create a correlation heatmap for numeric columns.
    """

    numeric_dataframe = dataframe.select_dtypes(
        include="number"
    )

    if numeric_dataframe.shape[1] < 2:
        return None

    correlation_matrix = numeric_dataframe.corr()

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=axis,
    )

    axis.set_title(
        "How the numeric columns move together",
        pad=16,
    )

    figure.tight_layout()

    return _figure_to_data_uri(figure)


def _dataset_summary(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Build a simple summary of the uploaded dataset.
    """

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    numeric_count = len(numeric_columns)
    missing_values = int(
        dataframe.isna().sum().sum()
    )

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "numeric_columns": numeric_count,
        "text_columns": int(
            dataframe.shape[1] - numeric_count
        ),
        "missing_values": missing_values,
    }


def _base_context(request) -> dict:
    """
    Values shared by all Project 1 templates.
    """

    return {
        "has_dataset": _has_uploaded_dataset(request),
        "dataset_name": request.session.get(
            SESSION_NAME_KEY,
            "",
        ),
    }


@require_http_methods(["GET", "POST"])
def index(request):
    """
    Upload page.

    Review and Train remain disabled until a valid CSV
    has been uploaded successfully.
    """

    form = DatasetUploadForm(
        request.POST or None,
        request.FILES or None,
    )

    if request.method == "POST" and form.is_valid():
        uploaded_file = form.cleaned_data["dataset"]

        
        _clear_uploaded_dataset(request)

        original_name = Path(
            uploaded_file.name
        ).name

        safe_name = (
            f"{uuid.uuid4().hex}_{original_name}"
        )

        destination = (
            _get_upload_directory() / safe_name
        )

        try:
            with destination.open("wb+") as output:
                for chunk in uploaded_file.chunks():
                    output.write(chunk)

        except OSError as exc:
            form.add_error(
                "dataset",
                f"The file could not be saved: {exc}",
            )

        else:
            request.session[SESSION_FILE_KEY] = str(
                destination
            )

            request.session[SESSION_NAME_KEY] = (
                original_name
            )

            request.session.modified = True

            try:
                _read_dataframe(request)

            except (
                FileNotFoundError,
                ValueError,
            ) as exc:
                destination.unlink(missing_ok=True)

                request.session.pop(
                    SESSION_FILE_KEY,
                    None,
                )

                request.session.pop(
                    SESSION_NAME_KEY,
                    None,
                )

                request.session.modified = True

                form.add_error(
                    "dataset",
                    str(exc),
                )

            else:
                messages.success(
                    request,
                    "Dataset uploaded successfully. "
                    "You can now review the data and train a model.",
                )

                return redirect(
                    "project1:visualize"
                )

    context = _base_context(request)

    context.update(
        {
            "form": form,
            "current_step": 1,
        }
    )

    return render(
        request,
        "project1/index.html",
        context,
    )

def _build_simple_graph_data(dataframe, max_points=250):
    """
    Prepare numerical data for the simple relationship graph.

    A maximum number of rows is used so that large datasets do not
    make the browser slow.
    """

    numeric_data = dataframe.select_dtypes(
        include="number",
    ).copy()

    numeric_columns = numeric_data.columns.tolist()

    if len(numeric_columns) < 2:
        return {
            "columns": [],
            "records": [],
            "default_x": "",
            "default_y": "",
        }

    # Limit the number of displayed points.
    if len(numeric_data) > max_points:
        numeric_data = numeric_data.sample(
            n=max_points,
            random_state=42,
        )

    # Convert missing and invalid values into JSON-compatible values.
    numeric_data = numeric_data.where(
        pd.notna(numeric_data),
        None,
    )

    return {
        "columns": numeric_columns,
        "records": numeric_data.to_dict(
            orient="records",
        ),
        "default_x": numeric_columns[0],
        "default_y": numeric_columns[1],
    }


@require_http_methods(["GET"])
def visualize(request):
    if not _has_uploaded_dataset(request):
        messages.warning(
            request,
            "Please upload a dataset before reviewing the data.",
        )
        return redirect("project1:index")

    try:
        dataframe = _read_dataframe(request)

    except Exception as error:
        messages.error(
            request,
            f"The dataset could not be opened: {error}",
        )
        return redirect("project1:index")

    preview = dataframe.head(10).to_dict(
        orient="records",
    )

    simple_graph = _build_simple_graph_data(
        dataframe,
        max_points=250,
    )

    return render(
        request,
        "project1/visualize.html",
        {
            "current_step": 2,
            "has_dataset": True,

            "dataset_name": request.session.get(
                SESSION_NAME_KEY,
                "Uploaded dataset",
            ),

            "summary": _dataset_summary(dataframe),
            "columns": dataframe.columns.tolist(),
            "preview": preview,

            # Simple graph information
            "numeric_columns": simple_graph["columns"],
            "graph_records": simple_graph["records"],
            "default_x_column": simple_graph["default_x"],
            "default_y_column": simple_graph["default_y"],
        },
    )

@require_http_methods(["GET", "POST"])
def mtrain(request):
    """
    Model training page.

    Direct access is blocked when no dataset exists.
    """

    if not _has_uploaded_dataset(request):
        messages.warning(
            request,
            "Please upload a dataset before training a model.",
        )

        return redirect("project1:index")

    try:
        dataframe = _read_dataframe(request)

    except (
        FileNotFoundError,
        ValueError,
    ) as exc:
        messages.error(
            request,
            str(exc),
        )

        return redirect("project1:index")

    form = TrainingForm(
        request.POST or None,
        columns=dataframe.columns.tolist(),
    )

    result = None

    if request.method == "POST":
        if form.is_valid():
            try:
                result = train_model(
                    dataframe=dataframe,
                    target_column=(
                        form.cleaned_data[
                            "target_column"
                        ]
                    ),
                    model_name=(
                        form.cleaned_data[
                            "model_name"
                        ]
                    ),
                    test_size_percent=(
                        form.cleaned_data[
                            "test_size"
                        ]
                    ),
                    normalize=(
                        form.cleaned_data[
                            "normalize"
                        ]
                    ),
                    fit_intercept=(
                        form.cleaned_data[
                            "fit_intercept"
                        ]
                    ),
                )

            except Exception as exc:
                messages.error(
                    request,
                    "We could not train the model. "
                    f"{exc}",
                )

        else:
            messages.error(
                request,
                "Please review the highlighted "
                "fields before training.",
            )

    context = _base_context(request)

    context.update(
        {
            "current_step": 3,
            "form": form,
            "result": result,
        }
    )

    return render(
        request,
        "project1/mtrain.html",
        context,
    )


def show_plot(request):
    """
    Redirect old plot URLs to the Review page.
    """

    if not _has_uploaded_dataset(request):
        messages.warning(
            request,
            "Please upload a dataset before viewing plots.",
        )

        return redirect("project1:index")

    return redirect("project1:visualize")


@require_http_methods(["GET", "POST"])
def reset(request):
    """
    Remove the current dataset and disable Review and Train again.
    """

    _clear_uploaded_dataset(request)

    messages.info(
        request,
        "The previous dataset was cleared. "
        "Upload a new dataset to continue.",
    )

    return redirect("project1:index")
