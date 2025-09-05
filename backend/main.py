import os
import cv2
import torch
import datetime
import logging
import pandas as pd
from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# FastAPI app
app = FastAPI(title="Box Detection Backend")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
processing_flag = False
processing_result = {}

YOLOV5_FULLPATH = os.path.abspath("yolov5")


# Model paths
MODEL_PATHS = {
    "Single Box": {
        "weights": os.path.abspath("backend/models/best5.pt"),
        "classes": ["box"]
    },
    "Multiple Box": {
        "weights": os.path.abspath("backend/models/best_demo2.pt"),
        "classes": ["box"]
    },
    "4_5_6 Box": {
        "weights": os.path.abspath("backend/models/best_demo_allbox.pt"),
        "classes": ["4box", "5box", "6box"]
    }
}

# Helper: load YOLOv5
def load_model(weights_path):
    return torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, source="github")

# Video processing function
def run_video_processing(video_url, supervisor_name, vehicle_no, selected_model):
    global processing_flag, processing_result

    model_info = MODEL_PATHS[selected_model]
    model = load_model(model_info["weights"])
    model.conf = 0.25
    model.iou = 0.45

    cap = None
    if video_url.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        cap = cv2.VideoCapture(video_url)
    else:
        cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        logger.error("Cannot open video")
        processing_flag = False
        return

    start_time = datetime.datetime.now()
    frame_count = 0
    total_detections = 0
    class_counts = {}

    while cap.isOpened() and processing_flag:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:  # Skip frames to save memory
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        for _, row in results.pandas().xyxy[0].iterrows():
            cls_name = row["name"]
            total_detections += 1
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    cap.release()
    end_time = datetime.datetime.now()

    processing_result = {
        "supervisor_name": supervisor_name,
        "vehicle_no": vehicle_no,
        "model_used": selected_model,
        "total_detections": total_detections,
        "classes_detected": class_counts,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "processing_duration_sec": (end_time - start_time).total_seconds(),
    }

    processing_flag = False

# Request body for /process-video
class VideoRequest(BaseModel):
    video_url: str
    supervisor_name: str
    vehicle_no: str
    selected_model: str

@app.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    video_url: str = Form(...),
    supervisor_name: str = Form(...),
    vehicle_no: str = Form(...),
    selected_model: str = Form(...)
):
    global processing_flag
    if processing_flag:
        return {"status": "busy", "message": "Processing already running"}

    processing_flag = True
    background_tasks.add_task(run_video_processing, video_url, supervisor_name, vehicle_no, selected_model)
    return {"status": "started", "message": "Processing started"}

@app.get("/results")
def get_results():
    if not processing_result:
        return {"status": "no_results"}
    return {"status": "done", "result": processing_result}

@app.get("/")
def root():
    return {"message": "Backend is running!"}
