import argparse
import mlflow
import mlflow.sklearn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to trained model")
    args = parser.parse_args()

    model = mlflow.sklearn.load_model(args.model)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_price_regressor",
        registered_model_name="used_cars_price_prediction_model",
    )

if __name__ == "__main__":
    main()
