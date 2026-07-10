# demos/ml_models.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#import time

class ModelTrainer:
    """
    Handles model training and evaluation with demo data (CLASSIFICATION)
    """
    
    def __init__(self):
        # Generate demo dataset for classification
        self.X_full, self.y_full = self._get_demo_data()
    
    def _get_demo_data(self):
        """Generate synthetic demo dataset for classification (binary classification)"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate binary target (0 or 1) for classification
        # Create a decision boundary based on features
        linear_combination = 2*X[:, 0] + 1.5*X[:, 1] - X[:, 2] + 0.5*X[:, 3]
        probability = 1 / (1 + np.exp(-linear_combination))  # Sigmoid function
        y = (probability > 0.5).astype(int)  # Binary classification
        
        return X, y
    
    def train_model(self, model_name, split_percentage):
        """Train the selected model with given split percentage"""
        
        
        # Split data based on user's preference
        test_size = 1 - (split_percentage / 100)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_full, self.y_full,
            test_size=test_size,
            random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select and train model
        model = self._get_model(model_name)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'Accuracy': round(accuracy, 4),
            #'Precision': round(precision, 4),
            #'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            #'Confusion Matrix': f"[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]"
        }
        
       
        
        # Model display names
        model_display_names = {
            'linear_regression': 'Logistic Regression',
            'decision_tree': 'Decision Tree Classifier',
            'knn': 'K-Nearest Neighbors Classifier',
            'svm': 'Support Vector Machine Classifier',
            'random_forest': 'Random Forest Classifier'
        }
        
        # Return results
        return {
            'model_name': model_display_names.get(model_name, model_name),
            'train_split': split_percentage,
            'test_split': 100 - split_percentage,
            'metrics': metrics,
        }
    
    def _get_model(self, model_name):
        """Return the appropriate classification model based on name"""
        models = {
            'linear_regression': LogisticRegression(random_state=42, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'svm': SVC(kernel='rbf', C=1.0, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }
        
        return models[model_name]