import asyncio
from io import BytesIO
from pathlib import Path

import numpy as np
from fastai.vision.all import PILImage, load_learner
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from src.eval_utils import resize_and_crop_center


MODEL_PICKEL_PATH = Path('models/model_pickle_fastai.pkl').absolute()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def setup_learner():
    learn = load_learner(MODEL_PICKEL_PATH)
    learn.dls.device = 'cpu'
    return learn


@app.on_event("startup")
async def startup_event():
    global learn
    tasks = [asyncio.ensure_future(setup_learner())]
    learn = (await asyncio.gather(*tasks))[0]


@app.post("/analyze")
async def analyze(image: bytes = File(...)):
    img = Image.open(BytesIO(image))
    img_cropped = resize_and_crop_center(img)
    img_cropped = PILImage(img_cropped)
    pred, *_ = learn.predict(img_cropped)
    pred = np.array(pred)
    resp = {'pred': pred.tolist()}
    return resp
