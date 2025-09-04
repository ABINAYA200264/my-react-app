import os
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

import cv2
import torch
import tempfile
import threading
import datetime
import numpy as np
import requests
import logging
from collections import deque
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve uploaded videos
UPLOAD_FOLDER = "uploaded_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/videos", StaticFiles(directory=UPLOAD_FOLDER), name="videos")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOLOV5_PATH = os.path.abspath("yolov5")

MODEL_PATHS = {
    "Single Box": {
        "weights": os.path.abspath("backend/best5.pt"),
        "classes": ["box"]
    },
    "Multiple Box": {
        "weights": os.path.abspath("backend/best_demo2.pt"),
        "classes": ["box"]
    },
    "4_5_6 Box": {
        "weights": os.path.abspath("backend/best_demo_allbox.pt"),
        "classes": ["4box", "5box", "6box"]
    }
}


CONF_THRESHOLD = 0.5
MODEL_CACHE = {}
processing_flag = False
processing_thread = None
processing_result = {}

def smooth_box(history):
    xs1, ys1, xs2, ys2 = zip(*history)
    return int(np.mean(xs1)), int(np.mean(ys1)), int(np.mean(xs2)), int(np.mean(ys2))

def get_centroid(box):
    xmin, ymin, xmax, ymax = box
    return int((xmin+xmax)/2), int((ymin+ymax)/2)

def is_crossing_line(prev_x, curr_x, line_x):
    return (prev_x < line_x and curr_x >= line_x) or (prev_x > line_x and curr_x <= line_x)

def get_model(selected_model):
    if selected_model in MODEL_CACHE:
        return MODEL_CACHE[selected_model]
    try:
        weights_path = MODEL_PATHS[selected_model]["weights"]
        model = torch.hub.load(YOLOV5_PATH, 'custom', path=weights_path, source='local', force_reload=False)
        model.eval()
        MODEL_CACHE[selected_model] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load model {selected_model}: {e}")
        return None

def send_count_email(count, to_email, start_time, end_time, supervisor_name, vehicle_no, class_counts):
    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = "Box Count Result"
    class_counts_str = "\n".join(f"{cls}: {cnt}" for cls, cnt in class_counts.items())
    body = f"""Supervisor: {supervisor_name}
Vehicle No: {vehicle_no}
Total Count: {count}
Counts per class:
{class_counts_str}
Start: {start_time}
End: {end_time}
"""
    message.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, message.as_string())
        server.quit()
        logger.info("Email sent successfully")
    except Exception as e:
        logger.error(f"Email sending failed: {e}")

def run_video_processing(video_url, supervisor_name, vehicle_no, selected_model, background_tasks):
    global processing_flag, processing_result
    if selected_model not in MODEL_PATHS:
        logger.error(f"Unknown model: {selected_model}")
        processing_flag = False
        return

    model = get_model(selected_model)
    if not model:
        processing_flag = False
        return

    classes = MODEL_PATHS[selected_model]["classes"]
    allowed_classes = [c.lower() for c in classes]
    object_trackers = {}
    box_history = {}
    next_object_id = 0
    class_counts = {}
    count = 0

    # Determine video source
    if video_url.startswith("http") or video_url.startswith("https"):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        response = requests.get(video_url, stream=True)
        for chunk in response.iter_content(chunk_size=8192):
            if chunk: tmp_file.write(chunk)
        tmp_file.close()
        cap = cv2.VideoCapture(tmp_file.name)
    elif video_url.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("Cannot open RTSP stream")
            processing_flag = False
            return
    else:
        video_path = video_url.replace("file://", "")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open local file")
            processing_flag = False
            return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_pos = frame_width // 2

    start_time = datetime.datetime.now()
    logger.info(f"Processing started at {start_time}")

    while cap.isOpened() and processing_flag:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections = results.pandas().xyxy[0]

        for _, row in detections.iterrows():
            cname = row['name'].lower()
            if row['confidence'] < CONF_THRESHOLD or cname not in allowed_classes:
                continue
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            cx, cy = get_centroid((xmin, ymin, xmax, ymax))

            matched_id = None
            min_dist = float('inf')
            for obj_id, prev_cx in object_trackers.items():
                dist = abs(prev_cx - cx)
                if dist < 50 and dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id
            if matched_id is None:
                matched_id = next_object_id
                next_object_id += 1
            object_trackers[matched_id] = cx
            if matched_id not in box_history: box_history[matched_id] = deque(maxlen=5)
            box_history[matched_id].append((xmin, ymin, xmax, ymax))

            if is_crossing_line(object_trackers[matched_id], cx, line_pos):
                increment = {"4box":4, "5box":5, "6box":6}.get(cname, 1)
                count += increment
                class_counts[cname] = class_counts.get(cname, 0) + increment
            object_trackers[matched_id] = cx

        processing_result = {
            "count": count,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": selected_model,
            "classes_detected": classes
        }

    cap.release()
    processing_flag = False
    logger.info(f"Processing finished. Total count: {count}")
    # Optional: send email
    # background_tasks.add_task(send_count_email, count, "your_email@gmail.com", start_time, datetime.datetime.now(), supervisor_name, vehicle_no, class_counts)

@app.post("/process-video")
async def process_video(
    file: UploadFile = File(None),
    video_url: str = Form(None),
    supervisor_name: str = Form(...),
    vehicle_no: str = Form(...),
    selected_model: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    global processing_flag, processing_thread, processing_result
    if processing_flag:
        raise HTTPException(status_code=400, detail="Processing already running")
    processing_flag = True
    processing_result = {}
    processing_thread = threading.Thread(target=run_video_processing,
                                         args=(video_url, supervisor_name, vehicle_no, selected_model, background_tasks))
    processing_thread.start()
    return {"status": "Processing started"}

@app.get("/processing-result")
async def processing_result_api():
    return JSONResponse({"status": "processing" if processing_flag else "done",
                         "result": processing_result})

@app.post("/stop-processing")
async def stop_processing():
    global processing_flag
    processing_flag = False
    return {"status": "Processing stopped"}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
