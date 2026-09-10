# Human-Centered Artificial Intelligence Projects (HCAI-PBL)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Django](https://img.shields.io/badge/Django-Web%20Framework-green)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![AI](https://img.shields.io/badge/Artificial-Intelligence-purple)

A collection of four **Human-Centered Artificial Intelligence (HCAI)** projects developed using **Django, Machine Learning, Explainable AI, Active Learning, and Human-AI Interaction techniques**.

The purpose of this repository is to develop interactive AI systems where users can understand, interact with, and influence machine learning models through web-based interfaces.

---

# 📌 Project Overview

This repository contains four Django-based applications:

| Project | Description |
|---|---|
| Project 1 | Automated Machine Learning Interface |
| Project 2 | Explainable Artificial Intelligence Interface |
| Project 3 | Active Learning for Learning-to-Defer |
| Project 4 | Preference Elicitation System |

---

# 📂 Repository Structure

```
HCAI-PBL/
│
├── home/                      # Main landing page
│
├── project1/                 # Automated ML Interface
│
├── project2/                 # Explainable AI
│
├── project3/                 # Active Learning & Learning-to-Defer
│
├── project4/                 # Preference Elicitation
│
├── static/                   # CSS, JavaScript and static files
│
├── media/                    # Uploaded files and generated plots
│
├── templates/                # HTML templates
│
├── manage.py                 # Django management file
│
├── requirements.txt          # Python dependencies
│
└── db.sqlite3                # Database
```

---

# 🚀 Project 1 — Automated Machine Learning Interface

## Objective

Develop a user-friendly interface that allows users to perform a complete supervised machine learning workflow.

## Features

✅ Upload CSV datasets  
✅ Dataset visualization  
✅ Data preprocessing  
✅ Train/test split  
✅ Machine learning model training  
✅ Model evaluation  
✅ Interactive ML workflow  


## Workflow

```
Dataset Upload
        ↓
Data Visualization
        ↓
Model Training
        ↓
Prediction
        ↓
Performance Evaluation
```

## Technologies

- Python
- Django
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

# 🔍 Project 2 — Explainable Artificial Intelligence Interface

## Objective

Build an interactive explainability system to understand machine learning predictions and model behaviour.

## Dataset

**Palmer Penguins Dataset**

Features include:

- Species
- Island
- Sex
- Year
- Bill length
- Bill depth
- Flipper length
- Body mass


## Implemented Methods

### Decision Tree Explainability

- Decision tree visualization
- Accuracy evaluation
- Model complexity analysis


### Regularization

- Lambda (λ) based complexity control
- Model comparison based on accuracy and complexity


### Logistic Regression Explanation

- Feature contribution analysis
- Prediction interpretation


### Counterfactual Explanation

Generate alternative examples showing:

> "What changes would make the model predict another class?"


### Feature Effect Visualization

Implemented:

- Partial Dependence Plot (PDP)
- Accumulated Local Effects (ALE)


## Technologies

- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

# 🤖 Project 3 — Active Learning for Learning-to-Defer

## Objective

Develop a Human-AI collaboration system where the AI model decides whether to:

- Make a prediction itself
- Defer the decision to a human expert


## Dataset

**AG News Dataset**

Task:

Classify news articles into different topics.


## Features

### Baseline Classifier

- Train classification model
- Evaluate prediction performance


### Simulated Expert

A simulated expert is created with:

- Different expertise areas
- Different prediction abilities


### Learning-to-Defer

The system learns:

```
Input
 ↓
AI Prediction OR
 ↓
Expert Decision
```


### Active Learning

Implemented:

- Efficient sample selection
- Expert querying strategy
- Learning expert competence


## Evaluation

Includes:

- Accuracy
- Deferral performance
- Model comparison

---

# 🎬 Project 4 — Preference Elicitation System

## Objective

Create a movie recommendation system that learns user preferences through interactive feedback.

## Dataset

**IMDb 5000 Movie Dataset**

---

## Implemented Features

✅ Movie feature extraction  
✅ Preference modelling  
✅ Interactive recommendation interface  
✅ User preference collection  
✅ Ranking-based preference learning  


## Preference Methods


### Method 1: Pairwise Comparison

User selects between two movies:

```
Movie A  VS  Movie B

        ↓

User preference
```


### Method 2: Ranking Interface

User ranks multiple movies:

```
1. Movie A
2. Movie B
3. Movie C
...
```


The system estimates user preferences based on collected interactions.

---

# 🛠 Technologies Used

## Programming Language

- Python


## Web Framework

- Django


## Machine Learning

- Scikit-learn
- NumPy
- Pandas


## Visualization

- Matplotlib


## Data Handling

- CSV
- SQLite Database


---

# ⚙️ Installation Guide

## 1. Clone Repository

```bash
git clone https://github.com/SanzinaHossain/HCAI-Projects.git
```

Navigate into project folder:

```bash
cd HCAI-PBL
```

---

# 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:


### Windows

```bash
venv\Scripts\activate
```


### Linux / macOS

```bash
source venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Database Migration

Run:

```bash
python manage.py migrate
```

---

# 5. Start Django Server

```bash
python manage.py runserver
```

Open browser:

```
http://127.0.0.1:8000/
```

---

# 🌐 Application Routes

| Application | URL |
|---|---|
| Home | `/` |
| Project 1 | `/project1/` |
| Project 2 | `/project2/` |
| Project 3 | `/project3/` |
| Project 4 | `/project4/` |

---

# 📊 Generated Outputs

The system generates:

- ML performance plots
- Decision tree visualizations
- PDP graphs
- ALE graphs
- Confusion matrices
- Uploaded datasets
- Recommendation outputs


Stored inside:

```
media/
```

---

# 📚 Datasets

## Project 1

- CSV datasets uploaded by users


## Project 2

- Palmer Penguins Dataset


## Project 3

- AG News Dataset


## Project 4

- IMDb 5000 Movie Dataset


---

# 👨‍💻 Author

Human-Centered Artificial Intelligence Course Project

Developed By Group 7


---

# 📄 License

This repository is created for educational and research purposes.
