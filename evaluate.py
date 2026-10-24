from pathlib import Path
import logging
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

from data_prep import load_data, split_data
from logger_config import logger_config

logger_config()


def evaluate():
    load_dotenv()
    data_path = os.getenv("DATA_PATH", "not found DATA_PATH")
    seed = os.getenv("SEED", "not found SEED")
    test_size = os.getenv("TEST_SIZE", "not found TEST_SIZE")

    for var in {data_path, seed, test_size}:
        if "not found" in var:
            print(f"ERROR: ENV variables not loaded! EXIT with code 1 {var}")
            exit(1)

    seed = int(seed)
    test_size = float(test_size)

    base_dir = Path(__file__).parent
    model_path = base_dir / "models" / "wine_classifier.joblib"
    results_path = base_dir / "wyniki.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. ")

    logging.info("Loading model...")
    model = joblib.load(model_path)

    logging.info("Loading data...")
    df = load_data(data_path)

    _, X_test, _, y_test = split_data(df, test_size=test_size, seed=seed)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    results = f"Ewaluacja modelu:\n\tAccuracy: {accuracy:.4f}\n\tClassification report:\n{report}"
    results_path.write_text(results, encoding="utf-8")
    logging.info("Accuracy: %.4f", accuracy)
    logging.info("Results saved in: %s", results_path)


if __name__ == "__main__":
    evaluate()
