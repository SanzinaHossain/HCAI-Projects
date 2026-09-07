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

    path(
        "simulated/",
        views.simulated_expert_view,
        name="simulated"
    ),

]