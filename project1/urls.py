from django.urls import path
from . import views

app_name = 'project1'

urlpatterns = [
    path('', views.index, name='index'),
    # path('upload/', views.upload_csv, name='upload'), 
    # path('plot/', views.generate_plot, name='plot'), 
    # path('generate-plot/', views.generate_plot_ajax, name='generate_plot_ajax'),

    #urls_for_modeltraining
    path('mtrain/', views.train, name='mtrain'),
#     path('train-model-form/', views.train, name='train_model_form'),
#     path("train/", views.train, name="train"),
]

