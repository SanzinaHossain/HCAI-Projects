from django.urls import path

from . import views


app_name = "project3"

urlpatterns = [

    # Main Active Learning page
    path(
        "",
        views.index,
        name="index"
    ),

    path("simulated/", views.simulated_expert_view, name="simulated"),
    path("learning-to-defer/", views.learning_to_defer_view, name="learning_to_defer"),
    path("active-learning/", views.active_learning_view, name="active_learning"),
]