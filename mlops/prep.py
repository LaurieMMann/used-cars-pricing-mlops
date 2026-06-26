import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--train_data", type=str, help="path to train data output")
    parser.add_argument("--test_data", type=str, help="path to test data output")
    args = parser.parse_args()

    mlflow.start_run()
    df = pd.read_csv(args.data)
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    mlflow.log_metric("train_rows", train_df.shape[0])
    mlflow.log_metric("test_rows", test_df.shape[0])

    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)

    mlflow.end_run()

if __name__ == "__main__":
    main()
