import pytest
import os
from data_prep import split_data, load_data
from dotenv import load_dotenv

load_dotenv()
SEED = os.getenv("SEED", "not found SEED")
DATA_PATH = os.getenv("DATA_PATH", "not found DATA_PATH")

for var in {SEED, DATA_PATH}:
    if "not found" in var:
        print(f"ERROR: ENV variables not loaded! EXIT with code 1 {var}")
        exit(1)

SEED = int(SEED)


def test_split_sizes():
    data = load_data(DATA_PATH)
    _, X_test, _, _ = split_data(data, test_size=0.2, seed=SEED)
    assert len(X_test) == pytest.approx(len(data) * 0.2, abs=1)


def test_sum_sets_sizes():
    data = load_data(DATA_PATH)
    X_train, X_test, _, _ = split_data(data, test_size=0.2, seed=SEED)
    assert len(X_train) + len(X_test) == len(data)


def test_wrong_input_load_data():
    with pytest.raises(FileNotFoundError):
        _ = load_data("This file doesn't exist")


def test_no_input_load_data():
    with pytest.raises(TypeError):
        _ = load_data()
