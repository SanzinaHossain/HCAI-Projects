from django.urls import path

from . import views

app_name = "project1"

urlpatterns = [
    path("", views.index, name="index"),
    path("visualize/", views.visualize, name="visualize"),
    path("train/", views.mtrain, name="mtrain"),
    path("show-plot/", views.show_plot, name="show_plot"),
    path("reset/", views.reset, name="reset"),
]
