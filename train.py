import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def main(max_depth, min_samples_split):
    # Load data
    data = pd.read_csv("data/Housing.csv")
    X = data.drop(columns=['median_house_value', 'ocean_proximity'])
    y = data['median_house_value']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
        
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log parameters, metrics, and model
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print("Done Training!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()
    
    # Start a new MLflow run
    with mlflow.start_run():
        main(args.max_depth, args.min_samples_split)