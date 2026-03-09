from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle

app = FastAPI()

# Load ML model
with open("model_iG.pkl", "rb") as f:
    model = pickle.load(f)

# HTML templates folder
templates = Jinja2Templates(directory="templates")

# CSS/Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(request: Request, area: float = Form(...)):
    area_val = np.array([[area]])
    prediction = model.predict(area_val)[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "area": area,
            "result": round(float(prediction), 2)
        }
    )