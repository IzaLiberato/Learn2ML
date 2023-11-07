import pickle
import re
from pathlib import Path

__version__ = "0.1.0"


BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/naive_bayes_iris_classifier-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)

classes = ['setosa', 'versicolor', 'virginica']


def predict_model(input):
    pred = model.predict([input])
    return classes[pred[0]]