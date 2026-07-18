from django import forms


MODEL_CHOICES = [
    ("logistic_regression", "Logistic Regression"),
    ("decision_tree", "Decision Tree"),
    ("random_forest", "Random Forest"),
    ("knn", "K-Nearest Neighbours"),
    ("svm", "Support Vector Machine"),
]


class DatasetUploadForm(forms.Form):
    dataset = forms.FileField(
        label="Choose a CSV file",
        widget=forms.ClearableFileInput(
            attrs={
                "accept": ".csv,text/csv",
                "class": "file-input",
            }
        ),
    )

    def clean_dataset(self):
        dataset = self.cleaned_data["dataset"]
        if not dataset.name.lower().endswith(".csv"):
            raise forms.ValidationError("Please upload a CSV file.")
        if dataset.size > 10 * 1024 * 1024:
            raise forms.ValidationError("The file is too large. Please keep it under 10 MB.")
        return dataset


class TrainingForm(forms.Form):
    target_column = forms.ChoiceField(
        label="What would you like to predict?",
        choices=[],
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    model_name = forms.ChoiceField(
        label="Choose a learning method",
        choices=MODEL_CHOICES,
        initial="logistic_regression",
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    test_size = forms.IntegerField(
        label="Testing data",
        min_value=10,
        max_value=40,
        initial=20,
        widget=forms.NumberInput(
            attrs={
                "type": "range",
                "min": "10",
                "max": "40",
                "step": "5",
                "class": "range-control",
            }
        ),
    )
    normalize = forms.BooleanField(
        label="Scale numeric values",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "toggle-input"}),
    )
    fit_intercept = forms.BooleanField(
        label="Allow a baseline value",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "toggle-input"}),
    )

    def __init__(self, *args, columns=None, **kwargs):
        super().__init__(*args, **kwargs)
        columns = columns or []
        self.fields["target_column"].choices = [
            ("", "Select the result column")
        ] + [(column, column) for column in columns]
