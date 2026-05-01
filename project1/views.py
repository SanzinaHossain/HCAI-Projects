from django.http import HttpResponse


def index(request):
    return HttpResponse("Project 1 : Automated Machine Learning.")