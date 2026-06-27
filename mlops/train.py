# CI/CD validation test - triggered manually on Jun 27, 2026
import argparse
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--model_output", type=str, help="path to save trained model")
    args = parser.parse_args()

    mlflow.start_run()

    train_df = pd.read_csv(os.path.join(args.train_data, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.test_data, "test.csv"))

    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    X_train = pd.get_dummies(X_train, columns=["Segment"], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=["Segment"], drop_first=True)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    mlflow.log_metric("mse", mse)

    mlflow.sklearn.save_model(model, args.model_output)

    mlflow.end_run()

if __name__ == "__main__":
    main()
