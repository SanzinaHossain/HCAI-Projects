from django.shortcuts import render


def index(request):
    context = {
        "title": "Project 3: Active Learning for Learning-to-Defer",
        "message": "Welcome to Project 3. This project will implement Active Learning and Learning-to-Defer on the AG News dataset."
    }

    return render(request, "project3/index.html", context)