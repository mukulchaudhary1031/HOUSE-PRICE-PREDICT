from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(
    request: Request,
    UNDER_CONSTRUCTION: float = Form(...),
    RERA: float = Form(...),
    BHK_NO: float = Form(...),
    SQUARE_FT: float = Form(...),
    READY_TO_MOVE: float = Form(...),
    RESALE: float = Form(...),
    LONGITUDE: float = Form(...),
    LATITUDE: float = Form(...),
    POSTED_BY: str = Form(...),
    BHK_OR_RK: str = Form(...),
    ADDRESS: str = Form(...)
):

    # Encoding categorical values
    posted_map = {"Owner": 0, "Builder": 1, "Dealer": 2}
    bhk_map = {"BHK": 0, "RK": 1}

    POSTED_BY = posted_map.get(POSTED_BY, 0)
    BHK_OR_RK = bhk_map.get(BHK_OR_RK, 0)

    # ADDRESS temporary encoding
    ADDRESS = 0


    data = pd.DataFrame({
    "UNDER_CONSTRUCTION":[UNDER_CONSTRUCTION],
    "RERA":[RERA],
    "BHK_NO.":[BHK_NO],
    "SQUARE_FT":[SQUARE_FT],
    "READY_TO_MOVE":[READY_TO_MOVE],
    "RESALE":[RESALE],
    "LONGITUDE":[LONGITUDE],
    "LATITUDE":[LATITUDE],
    "POSTED_BY":[POSTED_BY],
    "BHK_OR_RK":[BHK_OR_RK],
    "ADDRESS":[ADDRESS]
    })

    prediction = model.predict(data)[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": round(float(prediction), 2)
        }
    )