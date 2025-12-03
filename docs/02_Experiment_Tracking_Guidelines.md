# 📊 Experiment Tracking with MLflow - Implementation Guidelines

This guide walks you through implementing MLflow experiment tracking in your penguin classification project. You'll learn to track experiments, log parameters, metrics, and models, and control the tracking behavior using environment variables.

---

## 🎯 Learning Objectives

By the end of this guide, you will:
- ✅ Set up environment configuration using `.env` and `direnv`
- ✅ Integrate MLflow tracking into your training pipeline
- ✅ Log parameters, metrics, and models automatically
- ✅ Compare experiments in the MLflow UI
- ✅ Switch between local and MLflow storage modes

---

## 📋 Table of Contents

1. [Environment Setup with direnv](#1-environment-setup-with-direnv)
2. [MLflow Installation](#2-mlflow-installation)
3. [Creating the Configuration Module](#3-creating-the-configuration-module)
4. [Updating registry.py for MLflow](#4-updating-registrypy-for-mlflow)
5. [Integrating MLflow into Training](#5-integrating-mlflow-into-training)
6. [Testing Your Implementation](#6-testing-your-implementation)
7. [MLflow UI Usage](#7-mlflow-ui-usage)

---

## 1️⃣ Environment Setup with direnv

### Step 1.1: Create `.env` File

Create a `.env` file in your project root:

```bash
# .env
# MLflow Configuration

# Storage mode: "local" or "mlflow"
MODEL_STORAGE_MODE=mlflow

# MLflow tracking URI (local file system)
MLFLOW_TRACKING_URI=mlruns

# Experiment name
MLFLOW_EXPERIMENT_NAME=penguin_classification

# Model registry name
MODEL_NAME=penguin_classifier

# Local storage paths (used when MODEL_STORAGE_MODE=local)
LOCAL_MODEL_DIR=models
```

**💡 Explanation:**
- `MODEL_STORAGE_MODE`: Controls whether to use MLflow tracking or just save locally
- `MLFLOW_TRACKING_URI`: Where MLflow stores experiment data (use `mlruns` for local)
- `MLFLOW_EXPERIMENT_NAME`: Name of your experiment in MLflow
- `MODEL_NAME`: How your model will be registered in MLflow
- `LOCAL_MODEL_DIR`: Fallback directory for local-only mode

### Step 1.4: Create `.envrc` File

Create a `.envrc` file in your project root:

```bash
# .envrc
# Load environment variables from .env
dotenv
```

### Step 1.5: Allow direnv

Run this command in your project directory:

```bash
direnv allow .
```

You should see a message like:
```
direnv: loading ~/code/MLOps_week/.envrc
direnv: loading ~/code/MLOps_week/.env
```

**✅ Verification:**
```bash
echo $MODEL_STORAGE_MODE
# Should output: mlflow
```

### Step 1.6: Add to `.gitignore`

**Important:** Don't commit your `.env` file!

Add to `.gitignore`: 
```bash
# Environment variables
.env
.envrc

# MLflow
mlruns/
mlartifacts/
```

Create a `.env.example` for reference:
```bash
# .env.example
MODEL_STORAGE_MODE=local
MLFLOW_TRACKING_URI=mlruns
MLFLOW_EXPERIMENT_NAME=penguin_classification
MODEL_NAME=penguin_classifier
LOCAL_MODEL_DIR=models
```

---

## 2️⃣ MLflow Installation

### Step 2.1: Install MLflow

Add to your `pyproject.toml` (or install directly):

```bash
pip install mlflow
```

Or add to your `pyproject.toml`:
```toml
[project]
dependencies = [
    "mlflow>=2.9.0",
    # ... other dependencies
]
```

### Step 2.2: Verify Installation

```bash
mlflow --version
```

---

## 3️⃣ Creating the Configuration Module

### Step 3.1: Create `src/pengouins/config.py`

Create a new file to manage environment configuration:

```python
"""
Configuration module for managing environment variables and MLflow settings.
"""

import os
from pathlib import Path


class Config:
    """Configuration class for managing environment variables."""
    
    # Storage configuration
    MODEL_STORAGE_MODE = os.getenv("MODEL_STORAGE_MODE", "local")
    
    # MLflow configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "penguin_classification")
    MODEL_NAME = os.getenv("MODEL_NAME", "penguin_classifier")
    
    # Local storage configuration
    LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "models")
    
    @classmethod
    def use_mlflow(cls) -> bool:
        """Check if MLflow tracking is enabled."""
        return cls.MODEL_STORAGE_MODE.lower() == "mlflow"
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get the path for saving/loading models."""
        return Path(cls.LOCAL_MODEL_DIR) / f"{model_name}.pkl"
    
    @classmethod
    def display_config(cls):
        """Display current configuration."""
        print("=" * 50)
        print("📊 MLFLOW CONFIGURATION")
        print("=" * 50)
        print(f"Storage Mode: {cls.MODEL_STORAGE_MODE}")
        print(f"Use MLflow: {cls.use_mlflow()}")
        if cls.use_mlflow():
            print(f"MLflow Tracking URI: {cls.MLFLOW_TRACKING_URI}")
            print(f"Experiment Name: {cls.MLFLOW_EXPERIMENT_NAME}")
            print(f"Model Name: {cls.MODEL_NAME}")
        else:
            print(f"Local Model Directory: {cls.LOCAL_MODEL_DIR}")
        print("=" * 50)


# Display configuration on import (optional, for debugging)
if __name__ == "__main__":
    Config.display_config()
```

**💡 Key Features:**
- Reads environment variables with fallback defaults
- `use_mlflow()` method to check if MLflow is enabled
- `get_model_path()` for local storage paths
- `display_config()` to debug configuration

### Step 3.2: Test Configuration

Create a test script to verify your setup:

```python
# test_config.py
from src.pengouins.config import Config

Config.display_config()
print(f"\nMLflow enabled: {Config.use_mlflow()}")
```

Run it:
```bash
python test_config.py
```

---

## 4️⃣ Updating `registry.py` for MLflow

### Step 4.1: Add MLflow Imports

Update `src/pengouins/registry.py`:

```python
"""
Model registry for saving and loading models.
Supports both local storage and MLflow tracking.
"""

import pickle
import os
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from .config import Config


def save_model(model: Any, filepath: Optional[str] = None, model_name: Optional[str] = None):
    """
    Save a trained model to disk or MLflow.
    
    Args:
        model: The trained model to save
        filepath: Path to save the model (used in local mode)
        model_name: Name for the model in MLflow (used in mlflow mode)
    """
    if Config.use_mlflow():
        # MLflow mode: log model to current run
        if model_name is None:
            model_name = Config.MODEL_NAME
        
        print(f"💾 Logging model to MLflow as '{model_name}'...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        print(f"✅ Model logged to MLflow successfully!")
        
    else:
        # Local mode: save with pickle
        if filepath is None:
            filepath = Config.get_model_path(model_name or "model")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving model locally to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Model saved successfully!")


def load_model(filepath: Optional[str] = None, model_name: Optional[str] = None, 
               run_id: Optional[str] = None, version: Optional[int] = None) -> Any:
    """
    Load a model from disk or MLflow.
    
    Args:
        filepath: Path to load the model from (used in local mode)
        model_name: Name of the registered model in MLflow
        run_id: Specific run ID to load from MLflow
        version: Specific version to load from MLflow model registry
        
    Returns:
        The loaded model
    """
    if Config.use_mlflow():
        # MLflow mode: load from MLflow
        if model_name is None:
            model_name = Config.MODEL_NAME
        
        if run_id:
            # Load from specific run
            model_uri = f"runs:/{run_id}/model"
            print(f"📥 Loading model from MLflow run {run_id}...")
        elif version:
            # Load specific version from model registry
            model_uri = f"models:/{model_name}/{version}"
            print(f"📥 Loading model '{model_name}' version {version} from MLflow...")
        else:
            # Load latest version from model registry
            model_uri = f"models:/{model_name}/latest"
            print(f"📥 Loading latest model '{model_name}' from MLflow...")
        
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded successfully from MLflow!")
        return model
        
    else:
        # Local mode: load with pickle
        if filepath is None:
            filepath = Config.get_model_path(model_name or "model")
        
        print(f"📥 Loading model from {filepath}...")
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully!")
        return model


def save_preprocessor(preprocessor: Any, filepath: Optional[str] = None):
    """
    Save the preprocessor to disk or MLflow.
    
    Args:
        preprocessor: The fitted preprocessor to save
        filepath: Path to save (used in local mode)
    """
    if Config.use_mlflow():
        # Log as artifact in MLflow
        print(f"💾 Logging preprocessor to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=preprocessor,
            artifact_path="preprocessor"
        )
        print(f"✅ Preprocessor logged to MLflow!")
    else:
        # Save locally
        if filepath is None:
            filepath = Config.get_model_path("preprocessor")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving preprocessor to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump(preprocessor, f)
        print(f"✅ Preprocessor saved!")


def load_preprocessor(filepath: Optional[str] = None, run_id: Optional[str] = None) -> Any:
    """
    Load the preprocessor from disk or MLflow.
    
    Args:
        filepath: Path to load from (local mode)
        run_id: Run ID to load from (MLflow mode)
        
    Returns:
        The loaded preprocessor
    """
    if Config.use_mlflow():
        if run_id is None:
            raise ValueError("run_id is required when loading preprocessor from MLflow")
        
        model_uri = f"runs:/{run_id}/preprocessor"
        print(f"📥 Loading preprocessor from MLflow run {run_id}...")
        preprocessor = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Preprocessor loaded from MLflow!")
        return preprocessor
    else:
        if filepath is None:
            filepath = Config.get_model_path("preprocessor")
        
        print(f"📥 Loading preprocessor from {filepath}...")
        with open(filepath, "rb") as f:
            preprocessor = pickle.load(f)
        print(f"✅ Preprocessor loaded!")
        return preprocessor
```

**💡 Key Features:**
- Automatic mode detection using `Config.use_mlflow()`
- MLflow: Logs models as sklearn artifacts with model registry support
- Local: Falls back to pickle storage
- Support for versioning and run-specific loading
- Handles both models and preprocessors

---

## 5️⃣ Integrating MLflow into Training

### Step 5.1: Update `src/pengouins/model.py`

Add MLflow tracking to your training and evaluation:

```python
"""
Model training and evaluation with MLflow tracking support.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

from .config import Config


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a Logistic Regression model on the training data.
    
    Args:
        X_train: Preprocessed training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("🚀 Training Logistic Regression model...")
    
    # Log parameters if using MLflow
    if Config.use_mlflow():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("n_samples", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model parameters if using MLflow
    if Config.use_mlflow():
        mlflow.log_param("max_iter", model.max_iter)
        mlflow.log_param("solver", model.solver)
        mlflow.log_param("random_state", 42)
    
    print("✅ Model training completed!")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the model on the test data and return metrics.
    
    Args:
        model: Trained model
        X_test: Preprocessed test features
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("📊 Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted")
    }
    
    # Log metrics if using MLflow
    if Config.use_mlflow():
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
    
    # Print metrics
    print("\n" + "=" * 50)
    print("📈 MODEL EVALUATION RESULTS")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.capitalize()}: {metric_value:.4f}")
    print("=" * 50 + "\n")
    
    return metrics
```

**💡 What's New:**
- Automatic parameter logging (model type, hyperparameters, data info)
- Multiple metrics calculation (accuracy, precision, recall, F1)
- Automatic metric logging to MLflow
- Returns metrics dictionary for further analysis

### Step 5.2: Update `src/pengouins/data.py`

Add MLflow parameter logging for preprocessing:

```python
"""
Load and preprocess data with MLflow tracking support.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import mlflow

from .config import Config


# Module-level variable to store the preprocessor
_preprocessor = None


def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    print(f"📂 Loading data from {path}...")
    df = pd.read_csv(path)
    
    # Drop island column (it's the target in disguise)
    if "island" in df.columns:
        df = df.drop(columns=["island"])
    
    # Log data info if using MLflow
    if Config.use_mlflow():
        mlflow.log_param("data_path", path)
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_columns", len(df.columns))
    
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_X_y(df: pd.DataFrame, target_column: str, 
            target: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    if target:
        y = df.pop(target_column)
        X = df
        
        # Log target info if using MLflow
        if Config.use_mlflow():
            mlflow.log_param("target_column", target_column)
            mlflow.log_param("n_classes", len(y.unique()))
            mlflow.log_param("class_distribution", dict(y.value_counts()))
        
        return X, y
    else:
        return df, None


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """Split data into training and testing sets."""
    print(f"✂️  Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Log split info if using MLflow
    if Config.use_mlflow():
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
    
    print(f"✅ Data split completed!")
    return X_train, X_test, y_train, y_test


def preprocess_data(X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """Preprocess data using Pipeline and ColumnTransformer."""
    global _preprocessor
    
    if fit:
        print("🔧 Creating and fitting preprocessor...")
        
        # Numerical pipeline
        num_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        cat_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),   
            ('encoder', OneHotEncoder(sparse_output=False, drop="first"))
        ])
        
        # Combine pipelines
        _preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipe, make_column_selector(dtype_include="number")),
            ('cat', cat_pipe, make_column_selector(dtype_include=object))
        ])
        
        _preprocessor.fit(X)
        
        # Log preprocessing info if using MLflow
        if Config.use_mlflow():
            num_features = len(make_column_selector(dtype_include="number")(X))
            cat_features = len(make_column_selector(dtype_include=object)(X))
            mlflow.log_param("num_features", num_features)
            mlflow.log_param("cat_features", cat_features)
            mlflow.log_param("imputation_strategy_num", "median")
            mlflow.log_param("imputation_strategy_cat", "most_frequent")
            mlflow.log_param("scaling_method", "StandardScaler")
            mlflow.log_param("encoding_method", "OneHotEncoder")
        
        print("✅ Preprocessor fitted!")
    else:
        print("🔧 Transforming data with existing preprocessor...")
    
    X_transformed = _preprocessor.transform(X)
    return pd.DataFrame(X_transformed)


def get_preprocessor():
    """Get the fitted preprocessor."""
    return _preprocessor
```

**💡 What's New:**
- Data characteristics logged (rows, columns, classes)
- Split configuration logged
- Preprocessing parameters logged
- Class distribution tracked

---

## 6️⃣ Testing Your Implementation

### Step 6.1: Create Training Script

Create `main.py` (or update existing):

```python
"""
Main training script with MLflow integration.
"""

import mlflow
from src.pengouins.config import Config
from src.pengouins.data import (
    load_data, get_X_y, split_data, 
    preprocess_data, get_preprocessor
)
from src.pengouins.model import train_model, evaluate_model
from src.pengouins.registry import save_model, save_preprocessor


def main():
    """Main training pipeline."""
    
    # Display configuration
    Config.display_config()
    
    # Set up MLflow if enabled
    if Config.use_mlflow():
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        
        # Start MLflow run
        with mlflow.start_run(run_name="penguin_training"):
            print(f"\n🔬 MLflow Run ID: {mlflow.active_run().info.run_id}\n")
            run_pipeline()
    else:
        run_pipeline()


def run_pipeline():
    """Execute the training pipeline."""
    
    # 1. Load data
    df = load_data("data/pingouins.csv")
    
    # 2. Split into X and y
    X, y = get_X_y(df, target_column="species")
    
    # 3. Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 4. Preprocess
    X_train_processed = preprocess_data(X_train, fit=True)
    X_test_processed = preprocess_data(X_test, fit=False)
    
    # 5. Train model
    model = train_model(X_train_processed, y_train)
    
    # 6. Evaluate model
    metrics = evaluate_model(model, X_test_processed, y_test)
    
    # 7. Save model and preprocessor
    save_model(model, model_name=Config.MODEL_NAME)
    save_preprocessor(get_preprocessor())
    
    print("\n🎉 Training pipeline completed successfully!")
    
    if Config.use_mlflow():
        print(f"🔬 View results: mlflow ui")
        print(f"   Then open: http://localhost:5000")


if __name__ == "__main__":
    main()
```

### Step 6.2: Test Local Mode

First, test without MLflow:

```bash
# Update .env
MODEL_STORAGE_MODE=local

# Run training
python main.py
```

Expected output:
```
📊 MLFLOW CONFIGURATION
Storage Mode: local
Use MLflow: False
Local Model Directory: models
```

### Step 6.3: Test MLflow Mode

Now test with MLflow:

```bash
# Update .env
MODEL_STORAGE_MODE=mlflow

# Run training
python main.py
```

Expected output:
```
📊 MLFLOW CONFIGURATION
Storage Mode: mlflow
Use MLflow: True
MLflow Tracking URI: mlruns
Experiment Name: penguin_classification
```

### Step 6.4: Run Multiple Experiments

Try different configurations:

```python
# Experiment 1: Default settings
python main.py

# Experiment 2: Change test_size in split_data
# Update code: test_size=0.3
python main.py

# Experiment 3: Try different model
# Update model.py to use RandomForestClassifier
python main.py
```

---

## 7️⃣ MLflow UI Usage

### Step 7.1: Launch MLflow UI

```bash
mlflow ui
```

Or specify port:
```bash
mlflow ui --port 5001
```

### Step 7.2: Access the UI

Open your browser to:
```
http://localhost:5000
```

### Step 7.3: Explore Your Experiments

In the MLflow UI, you can:

1. **View All Runs:**
   - See all experiment runs in a table
   - Compare metrics across runs
   - Filter and search runs

2. **Run Details:**
   - Click on a run to see:
     - Parameters logged
     - Metrics logged
     - Artifacts (models, preprocessors)
     - Code version
     - Duration

3. **Compare Runs:**
   - Select multiple runs
   - Click "Compare"
   - See side-by-side parameter and metric comparison
   - Generate visualizations

4. **Model Registry:**
   - Navigate to "Models" tab
   - See registered models
   - View versions
   - Transition models between stages (Staging, Production)

### Step 7.4: Load Model from MLflow

```python
from src.pengouins.registry import load_model
from src.pengouins.config import Config

# Load latest model
model = load_model(model_name=Config.MODEL_NAME)

# Load specific version
model = load_model(model_name=Config.MODEL_NAME, version=1)

# Load from specific run
model = load_model(run_id="abc123def456")
```

---

## 🎯 Assignment Checklist

Complete these tasks to verify your implementation:

- [ ] Install and configure `direnv`
- [ ] Create `.env` file with all required variables
- [ ] Create `.envrc` and run `direnv allow`
- [ ] Add `.env` and `mlruns/` to `.gitignore`
- [ ] Create `.env.example` for documentation
- [ ] Install MLflow (`pip install mlflow`)
- [ ] Create `src/pengouins/config.py` with Config class
- [ ] Update `registry.py` with MLflow support
- [ ] Update `model.py` with parameter and metric logging
- [ ] Update `data.py` with data logging
- [ ] Create/update `main.py` with MLflow integration
- [ ] Test in local mode (`MODEL_STORAGE_MODE=local`)
- [ ] Test in MLflow mode (`MODEL_STORAGE_MODE=mlflow`)
- [ ] Run at least 3 experiments with different parameters
- [ ] Launch MLflow UI and explore your runs
- [ ] Compare experiments in the UI
- [ ] Successfully load a model from MLflow

---

## 🚀 Advanced Challenges

Once you've completed the basics, try these:

### Challenge 1: Add More Metrics
Add confusion matrix and classification report logging:

```python
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# In evaluate_model():
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')

if Config.use_mlflow():
    mlflow.log_figure(fig, "confusion_matrix.png")
```

### Challenge 2: Hyperparameter Tuning
Log multiple runs with different hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'max_iter': [100, 500, 1000]
}

for C in param_grid['C']:
    for max_iter in param_grid['max_iter']:
        with mlflow.start_run():
            model = LogisticRegression(C=C, max_iter=max_iter)
            # ... train and evaluate
```

### Challenge 3: Add Tags
Organize your experiments with tags:

```python
if Config.use_mlflow():
    mlflow.set_tag("model_type", "logistic_regression")
    mlflow.set_tag("dataset", "penguins")
    mlflow.set_tag("developer", "your_name")
    mlflow.set_tag("stage", "development")
```

### Challenge 4: Remote Tracking Server
Set up a remote MLflow server:

```bash
# In .env
MLFLOW_TRACKING_URI=http://your-mlflow-server.com:5000
```

---

## 📚 Additional Resources

- **MLflow Documentation:** https://mlflow.org/docs/latest/index.html
- **direnv Documentation:** https://direnv.net/
- **MLflow Tracking:** https://mlflow.org/docs/latest/tracking.html
- **MLflow Model Registry:** https://mlflow.org/docs/latest/model-registry.html

---

## ⚠️ Common Issues and Solutions

### Issue 1: direnv not loading
```bash
# Solution: Make sure you've added the hook to ~/.zshrc
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
source ~/.zshrc
direnv allow .
```

### Issue 2: MLflow can't find experiments
```bash
# Solution: Make sure MLFLOW_TRACKING_URI is set correctly
echo $MLFLOW_TRACKING_URI
# Should output: mlruns
```

### Issue 3: Model registry errors
```bash
# Solution: Create the experiment first
mlflow experiments create -n penguin_classification
```

### Issue 4: Port already in use
```bash
# Solution: Use a different port
mlflow ui --port 5001
```

---

## 🎓 Success Criteria

Your implementation is complete when:

- ✅ Environment variables load automatically with direnv
- ✅ Can switch between local and MLflow mode easily
- ✅ All experiments tracked in MLflow UI
- ✅ Parameters, metrics, and models logged correctly
- ✅ Can compare multiple runs in UI
- ✅ Can load models from MLflow registry
- ✅ Code is clean and well-organized

---

## 💡 Best Practices

1. **Experiment Naming:** Use descriptive run names
2. **Tagging:** Tag experiments for easy filtering
3. **Documentation:** Log important decisions as tags/notes
4. **Versioning:** Use semantic versioning for models
5. **Clean Up:** Regularly archive old experiments
6. **Collaboration:** Share MLflow tracking URI with team

---

Good luck with your MLflow implementation! 🚀🐧

Remember: The goal is to make your ML experiments **reproducible**, **trackable**, and **comparable**!
