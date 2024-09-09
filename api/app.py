from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse


import joblib
import numpy as np

from api.schemas import PredictRequest, Message
from data.species import species_map

# load the trained model
import os

model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
model = joblib.load(model_path)


# create instance of fastAPI
app = FastAPI(title="Prediction API - Iris")


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


# define the prediction endpoint
@app.post(
    path="/predict",
    response_model=Message,
    status_code=status.HTTP_200_OK,
    summary="Predição",
    description="Predição do tipo de flor com base das informações de petála e sepála",
    name="predição apí-iris",
)
def predict(request: PredictRequest):
    data = np.array(
        [
            [
                request.sepal_length,
                request.sepal_width,
                request.petal_length,
                request.petal_width,
            ]
        ]
    )

    prediction = model.predict(data)

    return {"prediction": species_map[prediction[0]]}
