from pathlib import Path

import joblib
import logging
import os
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data_prep import load_data, split_data
from logger_config import logger_config

logger_config()


def train():
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "not found DATA_PATH")
    seed = os.getenv("SEED", "not found SEED")
    test_size = os.getenv("TEST_SIZE", "not found TEST_SIZE")
    max_iter = os.getenv("MAX_ITER", "not found MAX_ITER")

    for var in {data_path, seed, test_size, max_iter}:
        if "not found" in var:
            print(f"ERROR: ENV variables not loaded! EXIT with code 1 {var}")
            exit(1)

    seed = int(seed)
    test_size = float(test_size)
    max_iter = int(max_iter)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, seed=seed)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=max_iter))
    ])
    model.fit(X_train, y_train)
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"wine_classifier.joblib"
    joblib.dump(model, model_path)
    logging.info(f"Model saved: {model_path}")


if __name__ == "__main__":
    train()
