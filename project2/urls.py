from django.urls import path
from . import views

app_name = 'project2'


urlpatterns = [
    path("", views.data_view, name="data"),
    path("tree/", views.tree_view, name="tree"),
    path("regularization/", views.regularization_view, name="regularization"),
    path("logistic/", views.logistic_view, name="logistic"),
    path("counterfactual/", views.counterfactual_view, name="counterfactual"),
]