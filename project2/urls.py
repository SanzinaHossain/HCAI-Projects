from django.urls import path
from . import views

app_name = "project2"

urlpatterns = [
    path("", views.home, name="project"),
    path("api/task1/", views.api_task1, name="api_task1"),
    path("api/model/", views.api_model, name="api_model"),
    path("api/examples/", views.api_examples, name="api_examples"),
    path("api/example/", views.api_example, name="api_example"),
    path("api/counterfactual/", views.api_counterfactual, name="api_counterfactual"),
    path("api/effects/", views.api_effects, name="api_effects"),
]
