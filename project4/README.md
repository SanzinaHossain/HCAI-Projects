# Project 4 - Preference Elicitation

This Django app implements **HCAI Project 4** as a self-contained `project4` folder. It compares two ways of learning a new user's movie preferences:

1. repeated two-movie choices (Bradley-Terry);
2. ranking sets of ten movies (Plackett-Luce extension).

The interface has been written so a **non-technical participant can understand and use it without knowing the mathematics**. Expandable help panels explain the study, technical terms, every movie field shown to the participant, questionnaire fields, held-out validation, and final agreement scores.

## Assignment tasks completed

- **Task 1:** a justified movie feature representation and extraction method is implemented in `services/feature_service.py`.
- **Task 2:** the Bradley-Terry pairwise model and a Plackett-Luce full-ranking extension are implemented in `services/preference_model.py`.
- **Task 3:** the downloadable PDF report contains a complete proposed user-study design: research question, hypotheses, within-subject design, recruitment, eligibility, counterbalancing plan, procedure, outcomes, timing, analysis, ethics, exclusions, and practical steps for a real launch.
- **Task 4:** the participant-facing study is implemented. The landing page contains both required actions: **Start the study** and **Download PDF report**.

The assignment does **not** require the user study to be conducted. This project does not invent participant results.

The submitted report is stored as the static file `project4/HCAI_Project_4_Report.pdf`. The **Project Report** button downloads this exact PDF; it is not generated from survey responses or from Python report-building code.

## Non-technical usability improvements

- visible **Project 4 Home** navigation on every study page plus a global **Main Home** link to `/`;
- expandable information/help toggles throughout the flow;
- movie-field definitions in plain language;
- progress bars and method/task numbering;
- same displayed movie metadata in both elicitation methods;
- ranking works with drag-and-drop **or Up/Down buttons** for touch/keyboard users;
- simple explanations of Bradley-Terry, Plackett-Luce, utility, features, and held-out validation;
- completion-time measurement for each elicitation method;
- final score explanations that explicitly warn against drawing a scientific conclusion from one session.

## Study design implemented in the demonstration

Each participant completes both methods. Order is randomly selected as pairwise-first or ranking-first for the demonstration. A real study should enforce equal AB/BA allocation across the final participant sample, as described in the PDF report.

Each method exposes 20 unique training movies:

- 10 pairwise tasks x 2 movies = 20 movies;
- 2 ranking tasks x 10 movies = 20 movies.

Eight additional held-out pairwise choices use 16 different movies for validation. Movies do not overlap between the two training methods and validation within the same session.

## Dataset

The assignment specifies the **IMDB 5000 Movie Dataset**. The dataset is already included at:

```text
project4/data/movie_metadata.csv
```

If it is removed, the app tries to download a public mirror on first use.

## Install Project 4 dependencies

From the project environment:

```bash
pip install -r project4/requirements-project4.txt
```

## Integrate with the existing HCAI-PBL Django project

In `pbl/settings.py`, add:

```python
"project4",
```

to `INSTALLED_APPS`.

In `pbl/urls.py`, make sure `include` is imported and add:

```python
path("project4/", include("project4.urls")),
```

Then run:

```bash
python manage.py check
python manage.py test project4
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/project4/
```

## Data-storage note

For the course demonstration, progress and responses are kept in the Django session. Before real participant recruitment, add deidentified persistent storage/export and controlled balanced AB/BA allocation, as described in the report.
