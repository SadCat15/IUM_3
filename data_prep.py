import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from logger_config import logger_config

logger_config()


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df.drop(columns=["Wine"]).to_numpy(dtype=np.float32)
    y = df["Wine"].to_numpy(dtype=np.float32)
    return train_test_split(X, y, test_size=test_size, random_state=seed)
