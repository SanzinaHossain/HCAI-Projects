
from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Sanzina Hossain", "matriculation": "676007"},
        {"name": "Md Samiul Hauque Chowdhury", "matriculation": "638038"},
        {"name": "Mehedi Hasan Talha", "matriculation": "672869"},
    ]
    
    projects = [
        {"name": "Project 1 : Automated Machine Learning", "url_name": "project1:index"},
        {"name": "Project 2 : Explainability", "url_name": "project2:project"},
        {"name": "Project 3 : Active Learning", "url_name": "project3:index"},
        {"name": "Project 4 : Preference Elicitation", "url_name": "project4:landing"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))