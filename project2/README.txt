HCAI PROJECT 2 - DJANGO INTEGRATION
==================================

This folder is designed to be copied directly into your existing HCAI-PBL Django project.

1. Replace your current D:\\Projects\\HCAI-PBL\\project2 folder with the project2 folder in this package.

2. Keep this line in pbl/urls.py:
       path("project2/", include("project2.urls")),

3. In pbl/settings.py, ensure project2 is in INSTALLED_APPS. For example:
       INSTALLED_APPS = [
           ...,
           "project2",
       ]
   If it is already there, do not add it twice.

4. In your activated virtual environment install/verify the required packages:
       pip install numpy pandas scikit-learn matplotlib

5. Start Django from D:\\Projects\\HCAI-PBL:
       python manage.py runserver

6. Open:
       http://127.0.0.1:8000/project2/

The app contains:
- urls.py (fixes the ModuleNotFoundError)
- Django views and JSON API endpoints
- Palmer Penguins dataset
- decision-tree and logistic-regression model logic
- counterfactual generator
- hand-written PDP and ALE calculations
- beginner explanations and field-information toggles
- Main Home navigation

No database migration is required for Project 2.


CSS/JS INTEGRATION NOTE
-----------------------
The Project 2 page embeds its CSS and JavaScript directly in templates/project2/index.html. This is intentional so the page works even when the parent Django project has a different STATIC_URL/STATICFILES setup. Copies are also retained in static/project2/ for normal Django static deployment if desired later.
