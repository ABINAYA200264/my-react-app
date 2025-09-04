import pathlib
# Patch: allow loading Windows-saved models on Linux
pathlib.WindowsPath = pathlib.PosixPath

import os
import smtplib
import threading
import tempfile
import cv2
import torch
import numpy as np
import datetime
import requests
import logging
from collections import deque
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder  # ✅ missing import fixed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOLOV5_FULLPATH = os.path.abspath("yolov5")

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

def smooth_box(history):
    xs1, ys1, xs2, ys2 = zip(*history)
    return int(np.mean(xs1)), int(np.mean(ys1)), int(np.mean(xs2)), int(np.mean(ys2))

def get_centroid(box):
    xmin, ymin, xmax, ymax = box
    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)
    return cx, cy

def is_crossing_line(prev_pos, current_pos, line_x):
    return (prev_pos < line_x and current_pos >= line_x) or \
           (prev_pos > line_x and current_pos <= line_x)

def send_count_email(count: int, to_email: str, start_time: datetime.datetime, end_time: datetime.datetime,
                    supervisor_name: str, vehicle_no: str, class_counts: dict):
    logger.info(f"Preparing to send email for count: {count}")
    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = "Box Count"
    class_counts_str = "\n".join(f"{cls}: {cnt}" for cls, cnt in class_counts.items())
    body = (f"Supervisor Name: {supervisor_name}\n"
            f"Vehicle Number: {vehicle_no}\n\n"
            f"Total count of Boxes : {count}\n\n"
            f"Counts per class:\n{class_counts_str}\n\n"
            f"Video processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Video processing ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    message.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.set_debuglevel(1)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, message.as_string())
        server.quit()
        logger.info("Email sent successfully!")
    except Exception as e:
        logger.error(f"Error sending email: {e}")


@app.get("/video/{filename}")
def get_video(filename: str):
    video_path = f"{filename}"
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


processing_flag = False
processing_thread = None
processing_result = {}
MODEL_CACHE = {}

def get_model(selected_model):
    if selected_model in MODEL_CACHE:
        return MODEL_CACHE[selected_model]
    config = MODEL_PATHS[selected_model]
    weights_path = config["weights"]
    try:
        model = torch.hub.load(
            YOLOV5_FULLPATH,
            'custom',
            path=weights_path,
            source='local',
            force_reload=False
        )
        model.eval()
        MODEL_CACHE[selected_model] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load model {selected_model}: {e}")
        return None

def run_video_processing(video_path, supervisor_name, vehicle_no, selected_model, background_tasks):
    global processing_flag, processing_result
    if selected_model not in MODEL_PATHS:
        logger.error(f"Unknown model: {selected_model}")
        processing_flag = False
        return
    model = get_model(selected_model)
    if model is None:
        processing_flag = False
        return
    config = MODEL_PATHS[selected_model]
    classes = config["classes"]
    count = 0
    object_trackers = {}
    next_object_id = 0
    class_counts = {}
    box_history = {}
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_position = frame_width // 2
        start_time = datetime.datetime.now()
        logger.info(f"Video processing started at: {start_time}")
        allowed_classes = [cls.lower() for cls in classes]
        while True:
            ret, frame = cap.read()
            if not ret or not processing_flag:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            detections = results.pandas().xyxy[0]

            cv2.line(frame, (line_position, 0), (line_position, frame_height), (0, 0, 255), 2)
            for _, row in detections.iterrows():
                cname = row['name'].lower()
                if row['confidence'] < CONF_THRESHOLD or cname not in allowed_classes:
                    continue
                xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                centroid = get_centroid((xmin, ymin, xmax, ymax))
                cx, cy = centroid
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
                if matched_id not in object_trackers:
                    object_trackers[matched_id] = cx
                if matched_id not in box_history:
                    box_history[matched_id] = deque(maxlen=5)
                box_history[matched_id].append((xmin, ymin, xmax, ymax))
                if is_crossing_line(object_trackers[matched_id], cx, line_position):
                    increment = 1
                    if cname == "4box":
                        increment = 4
                    elif cname == "5box":
                        increment = 5
                    elif cname == "6box":
                        increment = 6
                    count += increment
                    class_counts[cname] = class_counts.get(cname, 0) + increment
                object_trackers[matched_id] = cx
                sm_x1, sm_y1, sm_x2, sm_y2 = smooth_box(box_history[matched_id])
                color = (0, 255, 0)
                label = f"{cname} {row['confidence']:.2f} ID:{matched_id}"
                cv2.rectangle(frame, (sm_x1, sm_y1), (sm_x2, sm_y2), color, 2)
                cv2.putText(frame, label, (sm_x1, sm_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 3
            text = f"Count: {count}"
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x = (frame_width - text_width) // 2
            y = 50
            cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 255), thickness)
            details_text = ", ".join(f"{v} {k}" for k, v in class_counts.items())
            cv2.putText(frame, details_text, (x, y + 40), font, 1.0, (0, 255, 0), 2)
           
           
            logger.info(f"Video processing stopped by 'q' or stop flag, count={count}")
            
        cap.release()
        cv2.destroyAllWindows()
        end_time = datetime.datetime.now()
        logger.info(f"Video processing ended at: {end_time}")
        elapsed_time = end_time - start_time
        logger.info(f"Total boxes counted crossing line: {count}")
        processing_result = {
            "count": count,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_duration_sec": elapsed_time.total_seconds(),
            "model_used": selected_model,
            "classes_detected": classes
        }
        send_count_email(count, "abinayabi55@gmail.com", start_time, end_time, supervisor_name, vehicle_no, class_counts)
    except Exception as e:
        logger.error(f"Error in processing video: {e}")
        processing_flag = False


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

    tmp_video_path = None
    if file:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_file.write(await file.read())
        tmp_file.close()
        tmp_video_path = tmp_file.name
    elif video_url:
        tmp_video_path = video_url  # can be local Windows path, rtsp, or http stream
    else:
        processing_flag = False
        raise HTTPException(status_code=400, detail="No video input provided")

    processing_thread = threading.Thread(
        target=run_video_processing,
        args=(tmp_video_path, supervisor_name, vehicle_no, selected_model, background_tasks)
    )
    processing_thread.start()
    return {"status": "Processing started"}


@app.get("/processing-result")
async def processing_result_api():
    global processing_flag, processing_result
    if processing_flag:
        return JSONResponse({"status": "processing", "result": processing_result})
    else:
        return JSONResponse({"status": "done", "result": processing_result})


@app.post("/stop-processing")
async def stop_processing():
    global processing_flag
    if not processing_flag:
        return {"status": "Processing not running"}
    processing_flag = False
    return {"status": "Processing stopping"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({
            "detail": exc.errors(),
            "body": exc.body
        }),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
