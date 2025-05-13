from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Gerekirse buraya sadece mobil uygulamanın domainini yaz
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO modellerini yükle
model_sign = YOLO("models/yolov8nsign.pt")
model_light = YOLO("models/yolov8nlight.pt")
model_lane = YOLO("models/yolov8nlane.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Her modeli çalıştır
    result_sign = model_sign(img)[0].to_dict()
    result_light = model_light(img)[0].to_dict()
    result_lane = model_lane(img)[0].to_dict()

    return {
        "traffic_sign": result_sign,
        "traffic_light": result_light,
        "lane_detection": result_lane
    }

@app.get("/")
def root():
    return {"message": "YOLO API is running"}
