import csv
import io
import os
import numpy as np
from matplotlib import pyplot as plt


from django.conf import settings
from django.shortcuts import render
from .forms import CSVUploadForm
from django.http import HttpResponse, JsonResponse


# For model training
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ValidationError
import json
import logging
from .ml_models import ModelTrainer
logger = logging.getLogger(__name__)

def index(request):
    return HttpResponse("Welcome to Project 1!")


def upload_csv(request):
    result = None
    error = None

    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            decoded_file = file.read().decode('utf-8')
            io_string = io.StringIO(decoded_file)

            try:
                reader = csv.reader(io_string)
                numbers = []

                for row in reader:
                    for item in row:
                        try:
                            numbers.append(float(item.strip()))
                        except ValueError:
                            pass  # Skip non-numeric values

                if numbers:
                    result = sum(numbers) / len(numbers)
                else:
                    error = "No numeric values found in the CSV."
            except Exception as e:
                error = f"Error processing file: {str(e)}"
    else:
        form = CSVUploadForm()

    return render(request, 'project1/upload.html', {
        'form': form,
        'result': result,
        'error': error
    })


def save_plot(filename):
    image_path = os.path.join(settings.MEDIA_ROOT, filename)
    
    x = np.random.rand(10)
    y = np.random.rand(10)
    plt.figure()
    plt.scatter(x, y)
    plt.savefig(image_path)
    
    image_url = settings.MEDIA_URL + filename
    return image_url



def generate_plot(request):
    filename = 'myplot.png'
    image_url = save_plot(filename)
    return render(request, 'project1/show_plot.html', {'image_url': image_url})


def generate_plot_ajax(request):
    if request.method == "POST":
        filename = 'myplot.png'
        image_url = save_plot(filename)
        return JsonResponse({'image_url': image_url})



# Model training views

def train_page(request):
    return render(request, "project1/mtrain.html")

def train_model_form_view(request):
    """Handle traditional form submission"""
    if request.method == 'POST':
        model_name = request.POST.get('model')
        split_percentage = int(request.POST.get('split_percentage', 80))
        
        trainer = ModelTrainer()
        results = trainer.train_model(model_name, split_percentage)
        
        return render(request, 'project1/mtrain.html', {'results': results})
    
    return render(request, 'project1/mtrain.html')