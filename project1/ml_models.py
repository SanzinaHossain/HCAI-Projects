# demos/ml_models.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

class ModelTrainer:
    """
    Handles model training and evaluation with uploaded CSV data (CLASSIFICATION)
    """
    
    def __init__(self):
        # Initialize with no data
        self.X_full = None
        self.y_full = None
        self.data = None
        self.target_column = None
    
    def load_data(self, file_path):
        """Load dataset from CSV file path"""
        self.data = pd.read_csv(file_path)
        print(f"Dataset loaded: {self.data.shape}")
        print(f"Columns available: {list(self.data.columns)}")
        return self.data
    
    def prepare_data(self, target_column):
        """Prepare features and target from loaded data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.target_column = target_column
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Handle categorical features (convert to numeric)
        X = self._encode_categorical_features(X)
        
        # Handle categorical target
        if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"Target classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
        
        # Convert to numpy arrays
        self.X_full = X.values.astype(np.float32)
        self.y_full = np.array(y).astype(np.int32)
        
        print(f"Prepared data: {self.X_full.shape[0]} samples, {self.X_full.shape[1]} features")
        return self.X_full, self.y_full
    
    def _encode_categorical_features(self, X):
        """Convert categorical columns to numeric using one-hot encoding"""
        # Select categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            print(f"Encoding categorical columns: {list(categorical_cols)}")
            # One-hot encode categorical columns
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Handle any remaining non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        return X
    
    def train_model(self, model_name, split_percentage, hyperparams=None):
        """Train the selected model with given split percentage and hyperparams"""
    
        if self.X_full is None or self.y_full is None:
            raise ValueError("No data prepared. Call prepare_data() first.")
    
        if hyperparams is None:
            hyperparams = {}

        test_size = 1 - (split_percentage / 100)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_full, self.y_full,
            test_size=test_size,
            random_state=42,
            stratify=self.y_full
        )
    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Pass hyperparams to model
        model = self._get_model(model_name, hyperparams)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        accuracy  = accuracy_score(y_test, y_pred)
        unique_classes = np.unique(self.y_full)
        avg = 'binary' if len(unique_classes) == 2 else 'weighted'
        precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
        recall    = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1        = f1_score(y_test, y_pred, average=avg, zero_division=0)

        metrics = {
            'Accuracy':  round(accuracy,  4),
            'Precision': round(precision, 4),
            'Recall':    round(recall,    4),
            'F1-Score':  round(f1,        4),
        }

        model_display_names = {
            'linear_regression': 'Logistic Regression',
            'decision_tree':     'Decision Tree Classifier',
            'knn':               'K-Nearest Neighbors',
            'svm':               'Support Vector Machine',
            'random_forest':     'Random Forest Classifier',
        }

        return {
            'model_name':   model_display_names.get(model_name, model_name),
            'train_split':  split_percentage,
            'test_split':   100 - split_percentage,
            'metrics':      metrics,
            'num_samples':  self.X_full.shape[0],
            'num_features': self.X_full.shape[1],
            'num_classes':  len(unique_classes),
        }


    def _get_model(self, model_name, hyperparams):
        """Build model using user-supplied hyperparams with safe fallback defaults."""

        def get(key, default, cast=None):
            val = hyperparams.get(key, default)
            if cast:
                try:
                    return cast(val)
                except (ValueError, TypeError):
                    return default
            return val

        if model_name == 'linear_regression':
            return LogisticRegression(
                fit_intercept = get('fit_intercept', 'true') == 'true',
                random_state  = 42,
                max_iter      = 1000,
            )

        elif model_name == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth         = get('max_depth',         5,     int),
                min_samples_split = get('min_samples_split', 2,     int),
                min_samples_leaf  = get('min_samples_leaf',  1,     int),
                criterion         = get('criterion',         'gini'),
                random_state      = 42,
            )

        elif model_name == 'knn':
            return KNeighborsClassifier(
                n_neighbors = get('n_neighbors', 5,           int),
                weights     = get('weights',     'uniform'),
                metric      = get('metric',      'euclidean'),
                algorithm   = get('algorithm',   'auto'),
            )

        elif model_name == 'svm':
            return SVC(
                C            = get('C',      1,     int),
                kernel       = get('kernel', 'rbf'),
                gamma        = get('gamma',  'scale'),
                degree       = get('degree', 3,     int),
                random_state = 42,
            )

        elif model_name == 'random_forest':
            max_features_val = get('max_features', 'sqrt')
            if max_features_val == 'none':
                max_features_val = None

            return RandomForestClassifier(
                n_estimators      = get('n_estimators',      100, int),
                max_depth         = get('max_depth',         10,  int),
                min_samples_split = get('min_samples_split', 2,   int),
                max_features      = max_features_val,
                bootstrap         = get('bootstrap', 'true') == 'true',
                random_state      = 42,
            )

        raise ValueError(f"Unknown model: {model_name}")