from fastapi import FastAPI
from pydantic import BaseModel
from model.model import predict_model
from model.model import __version__ as model_version


from typing import List, Optional


app = FastAPI()


class TextIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionOut(BaseModel):
    species: str



@app.get("/")
def home():
    return {"server_status": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    sepal_length = payload.sepal_length
    sepal_width = payload.sepal_width
    petal_length = payload.petal_length
    petal_width = payload.petal_width
    input = [sepal_length, sepal_width, petal_length, petal_width]
    species = predict_model(input)
    return {"species": species}