
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import loguru

from pengouins.registry import save_model


def train_model(  X_train : pd.DataFrame
                , y_train: pd.Series) -> LogisticRegression:
    """Train a model on the training data."""
    loguru.logger.info("Training Model...")
    logi = LogisticRegression()
    logi.fit(X_train, y_train)
    save_model(logi,"logistic_reg")
    return logi
    

def evaluate_model( model
                    , X_test: pd.DataFrame
                    , y_test: pd.Series) -> float:
    """Evaluate the model on the test data and return accuracy."""
    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_pred,y_test),3)
    loguru.logger.info(f"Model accuracy : {acc}")
    return acc