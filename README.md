# Human-Centered Artificial Intelligence Projects

This repository contains course projects developed for the Human-Centered Artificial Intelligence course using Django and machine learning techniques.

---

# Project Structure

```text
HCAI-PBL/
│
├── home/
├── project1/
├── project2/
├── static/
├── media/
├── manage.py
```

---

# Project 1 — Supervised Learning Interface

Implemented functionalities:

- CSV dataset upload
- Dataset visualization
- Machine learning model training
- Train/test split
- Model evaluation
- Interactive web interface using Django

Main objective:

Develop a user-friendly interface for supervised machine learning workflows. :contentReference[oaicite:0]{index=0}

---

# Project 2 — Explainable AI Interface

Implemented functionalities:

- Decision Tree visualization
- Regularization using λ optimization
- Logistic Regression explainability
- Counterfactual explanations
- PDP (Partial Dependence Plot)
- ALE (Accumulated Local Effects)

Dataset used:

- Palmer Penguins Dataset

Main objective:

Explore interpretable and human-centered machine learning methods. :contentReference[oaicite:1]{index=1}

---

# Technologies Used

- Python
- Django
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- HTML/CSS

---

# Installation

Clone repository:

```bash
git clone <https://github.com/SanzinaHossain/HCAI-Projects.git>
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
python manage.py runserver
```

Open browser:

```text
http://127.0.0.1:8000/
```

