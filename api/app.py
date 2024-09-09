from asyncio import to_thread

from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse
import joblib
import os
from api.schemas import PredictRequest, Message
from data.species import species_map

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
model = joblib.load(model_path)

# Create instance of FastAPI
app = FastAPI(title="Prediction API - Iris")

@app.get("/")
async def read_root():
    return RedirectResponse(url="/docs")

# Define the prediction endpoint
@app.post(
    path="/predict",
    response_model=Message,
    status_code=status.HTTP_200_OK,
    summary="Predição",
    description="Predição do tipo de flor com base das informações de pétala e sépala",
    name="predição apí-iris",
)
async def predict(request: PredictRequest):
    data = [
        [
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]
    ]

    # Execute model prediction
    prediction = await to_thread(model.predict, data)

    return {"prediction": species_map[prediction[0]]}
