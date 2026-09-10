# Add Project 4 to the existing HCAI-PBL project

Only two edits outside this folder are required.

## 1. pbl/settings.py
Add `"project4",` inside `INSTALLED_APPS`.

## 2. pbl/urls.py
Ensure this import exists:

```python
from django.urls import include, path
```

Add this URL pattern:

```python
path("project4/", include("project4.urls")),
```

Nothing in project1, project2, or project3 needs to be replaced.
