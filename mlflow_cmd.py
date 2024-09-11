import argparse

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# set tracking uri
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Compare RandomForests")


def load_data():
    """load, prep and return Data"""
    train_df = pd.read_csv("diabetes_train_usable.csv")
    test_df = pd.read_csv("diabetes_test_usable.csv")

    X_train = train_df.drop("diabetes", axis=1)
    X_test = test_df.drop("diabetes", axis=1)
    y_train = train_df["diabetes"]
    y_test = test_df["diabetes"]

    return X_train, X_test, y_train, y_test 


def evaluate_model(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    return acc, f1_weighted, precision_weighted, recall_weighted


# RandomForestClassifer
# n_estimators, examples [50, 100, 150, 200]
# criterion, examples ["gini", "entropy", "log_loss"]
# max_depth, examples [None, 5, 10]
# min_samples_split, examples [1, 2, 5, 10]
# min_samples_leaf, examples [1, 2, 5, 10]
# max_features, examples ["sqrt", "log2", None]


def main(params: dict):
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run() as run:

        mlflow.log_params(params=params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc, f1, precision, recall = evaluate_model(y_pred=y_pred, y_true=y_test)

        mlflow.log_metric("Accuracy", round(acc * 100, 2))
        mlflow.log_metric("F1-Score", round(f1 * 100, 2))
        mlflow.log_metric("Precision", round(precision * 100, 2))
        mlflow.log_metric("Recall", round(recall * 100, 2))
        
    mlflow.end_run()




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--n_estimators", "-n_est", type=int, default=100)
    args.add_argument("--criterion", "-crit", type=str, default="gini")
    args.add_argument("--max_depth", "-m_dep", type=int or None, default=None)
    args.add_argument("--min_samples_split", "-m_split", type=int or float, default=2)
    args.add_argument("--min_samples_leaf", "-m_leaf", type=int or float, default=1)
    args.add_argument("--max_features", "-m_fea", type=str or None, default="sqrt")
    parsed_args = args.parse_args()
    params = vars(parsed_args)
    # parsed_args.param1
    main(params=params)