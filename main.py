
from pengouins.data import load_data, get_X_y, split_data, preprocess_data
from pengouins.model import train_model, evaluate_model
from pengouins.registry import save_model, load_model

def main():
    # Load data
    df = load_data("data/penguins.csv")

    # # Prepare features and target
    # X, y = get_X_y(df, target_column="species")

    # # Split data
    # X_train, X_test, y_train, y_test = split_data(X, y)

    # # Preprocess data
    # X_train_preproc = preprocess_data(X_train, fit=True)
    # X_test_preproc = preprocess_data(X_test, fit=False)

    # # Train model
    # model = train_model(X_train_preproc, y_train)

    # # Evaluate model
    # accuracy = evaluate_model(model, X_test_preproc, y_test)
    # print(f"Model accuracy: {accuracy}")

    # # Save model
    # save_model(model, "models/logistic_regression_model.pkl")