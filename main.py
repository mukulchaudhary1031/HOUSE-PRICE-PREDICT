from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle

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
    POSTED_BY: float = Form(...),
    UNDER_CONSTRUCTION: float = Form(...),
    RERA: float = Form(...),
    BHK_NO: float = Form(...),
    BHK_OR_RK: float = Form(...),
    SQUARE_FT: float = Form(...),
    READY_TO_MOVE: float = Form(...),
    RESALE: float = Form(...),
    ADDRESS: float = Form(...),
    LONGITUDE: float = Form(...),
    LATITUDE: float = Form(...)
):

    data = np.array([[
        POSTED_BY,
        UNDER_CONSTRUCTION,
        RERA,
        BHK_NO,
        BHK_OR_RK,
        SQUARE_FT,
        READY_TO_MOVE,
        RESALE,
        ADDRESS,
        LONGITUDE,
        LATITUDE
    ]])

    prediction = model.predict(data)[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": round(float(prediction), 2)
        }
    )