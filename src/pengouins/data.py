"""
Load and preprocess data.
"""

import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import loguru
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from pengouins.registry import save_model, load_model

logger = loguru.logger

def load_data(path: str) -> pd.DataFrame:
    """Load data from seaborn data,
    put it in cache and return a DataFrame."""
    
    csv_path = Path("data", path)
    
    if csv_path.exists():
        logger.info(f"Loading data from cache at {csv_path}")
        return pd.read_csv(csv_path)
    
    pingouins = sns.load_dataset(path)
    if not os.path.exists("data"):
        os.makedirs("data")
    # Drop the island column becuase it's exactly the target variable in disguise
    pingouins.drop(columns=["island"], inplace=True)
    pingouins.to_csv(csv_path, index=False)

    return pingouins

def get_X_y(
    df: pd.DataFrame, target_column: str, target:bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    if not target_column in df.columns : 
        raise Exception("Missing Target in the Dataset")
    X = df.copy()
    y = X.pop(target_column) # NB: pop remove the col and returns it 
    return X,y
    

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def preprocess_data(X: pd.DataFrame
                    ,fit = True) -> pd.DataFrame:
    """Preprocess data: handle missing values, encode categorical variables, scale numerical features."""

    if fit :
        logger.info("Initializing Preprocessor ...")
        num_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),   
            ('encoder', OneHotEncoder(sparse_output=False, drop="first"))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipe, make_column_selector(dtype_include="number")),
            ('cat', cat_pipe, make_column_selector(dtype_include=object))
        ]).set_output(transform="pandas")

        preprocessor.fit(X)
        
        save_model(preprocessor,"preprocessor")
    else :   
        preprocessor = load_model("preprocessor")
        
    X_preproc = preprocessor.transform(X)
    return X_preproc
    


if __name__ == "__main__": 
    data = load_data("penguins") 
    X,y = get_X_y(data,"species")
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_preproc = preprocess_data(X_train, fit=True)
    X_test_preproc  = preprocess_data(X_test, fit=False)