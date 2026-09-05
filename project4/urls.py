from django.urls import path
from . import views

app_name='project4'
urlpatterns=[
 path('',views.landing,name='landing'), path('study/start/',views.start,name='start'), path('study/consent/',views.consent,name='consent'),
 path('study/task/',views.task,name='task'), path('study/survey/',views.survey,name='survey'), path('study/validation/',views.validation,name='validation'),
 path('study/complete/',views.complete,name='complete'), path('study/reset/',views.reset,name='reset'), path('report/',views.report_pdf,name='report'),
]
