# import pathlib
# pathlib.PosixPath = pathlib.WindowsPath


# from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.exceptions import RequestValidationError
# from fastapi.encoders import jsonable_encoder
# from fastapi import status

# import threading
# import torch
# import cv2
# import os
# import tempfile
# import requests
# from torchvision import transforms
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import smtplib
# import logging
# import datetime
# import numpy as np
# from collections import deque

# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()# from dotenv import load_dotenv

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)


# app = FastAPI()


# origins = [
#     "http://localhost:3000",
#     "http://localhost:3001",
# ]


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# YOLOV5_FULLPATH = "D:/Vchanel/Box_detection_web/yolov5"


# MODEL_PATHS = {
#     "Single Box": {
#         "weights": "D:/Vchanel/Box_detection_web/backend/best5.pt",
#         "classes": ["box"]
#     },
#     "Multiple Box": {
#         "weights": "D:/Vchanel/Box_detection_web/backend/best_demo2.pt",
#         "classes": ["box"]
#     },
#     "4_5_6 Box": {
#         "weights": "D:/Vchanel/Box_detection_web/backend/best_demo_allbox.pt",
#         "classes": ["4box", "5box", "6box"]
#     }
# }


# CONF_THRESHOLD = 0.5



# def smooth_box(history):
#     xs1, ys1, xs2, ys2 = zip(*history)
#     return int(np.mean(xs1)), int(np.mean(ys1)), int(np.mean(xs2)), int(np.mean(ys2))



# def get_centroid(box):
#     xmin, ymin, xmax, ymax = box
#     cx = int((xmin + xmax) / 2)
#     cy = int((ymin + ymax) / 2)
#     return cx, cy



# def is_crossing_line(prev_pos, current_pos, line_y):
#     return prev_pos < line_y and current_pos >= line_y



# def send_count_email(count: int, to_email: str, start_time: datetime.datetime, end_time: datetime.datetime,
#                      supervisor_name: str, vehicle_no: str, class_counts: dict):
#     logger.info(f"Preparing to send email for count: {count}")


#     sender_email = os.getenv("EMAIL_ADDRESS")  # Your Gmail address
#     sender_password = os.getenv("EMAIL_PASSWORD")  # Your Gmail app password
#     smtp_server = "smtp.gmail.com"
#     smtp_port = 587


#     message = MIMEMultipart()
#     message["From"] = sender_email
#     message["To"] = to_email
#     message["Subject"] = "Box Count"


#     class_counts_str = "\n".join(f"{cls}: {cnt}" for cls, cnt in class_counts.items())


#     body = (f"Supervisor Name: {supervisor_name}\n"
#             f"Vehicle Number: {vehicle_no}\n\n"
#             f"Total count of Boxes : {count}\n\n"
#             f"Counts per class:\n{class_counts_str}\n\n"
#             f"Video processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
#             f"Video processing ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


#     message.attach(MIMEText(body, "plain"))


#     try:
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.set_debuglevel(1)
#         server.starttls()
#         server.login(sender_email, sender_password)
#         server.sendmail(sender_email, to_email, message.as_string())
#         server.quit()
#         logger.info("Email sent successfully!")
#     except Exception as e:
#         logger.error(f"Error sending email: {e}")



# @app.get("/video/{filename}")
# def get_video(filename: str):
#     video_path = f"{filename}"
#     if not os.path.isfile(video_path):
#         raise HTTPException(status_code=404, detail="Video not found")
#     return FileResponse(video_path, media_type="video/mp4")



# processing_flag = False
# processing_thread = None
# processing_result = {}

# def run_video_processing(file, video_url, supervisor_name, vehicle_no, selected_model, background_tasks):
#     global processing_flag, processing_result

#     if selected_model not in MODEL_PATHS:
#         logger.error(f"Unknown model: {selected_model}")
#         processing_flag = False
#         return

#     config = MODEL_PATHS[selected_model]
#     weights_path = config["weights"]
#     classes = config["classes"]

#     count = 0
#     object_trackers = {}
#     next_object_id = 0
#     class_counts = {}

#     box_history = {}
#     box_confidence = {}
#     active_objects = {}
#     object_states = {}  # 0=outside, 1=inside
#     region_width = 40

#     try:
#         if video_url:
#             if video_url.startswith("file://") or os.path.exists(video_url):
#                 video_path = video_url.replace("file://", "")
#                 if not os.path.exists(video_path):
#                     logger.error(f"Local file not found: {video_path}")
#                     processing_flag = False
#                     return
#             else:
#                 tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 response = requests.get(video_url, stream=True)
#                 response.raise_for_status()
#                 for chunk in response.iter_content(chunk_size=8192):
#                     if chunk:
#                         tmp_file.write(chunk)
#                 tmp_file.close()
#                 video_path = tmp_file.name
#         else:
#             logger.error("No video file or URL provided")
#             processing_flag = False
#             return


#         model = torch.hub.load(
#             YOLOV5_FULLPATH,
#             'custom',
#             path=weights_path,
#             source='local',
#             force_reload=False
#         )
#         model.eval()


#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         line_position = frame_width // 2   # vertical line position


#         start_time = datetime.datetime.now()
#         logger.info(f"Video processing started at: {start_time}")


#         allowed_classes = [cls.lower() for cls in classes]


#         while True:
#             ret, frame = cap.read()
#             if not ret or not processing_flag:
#                 break


#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = model(frame_rgb)
#             detections = results.pandas().xyxy[0]


#             # Draw vertical red line for count zone
#             cv2.line(frame, (line_position, 0), (line_position, frame_height), (0, 0, 255), 2)


#             region_start = line_position - region_width
#             region_end = line_position + region_width
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (region_start, 0), (region_end, frame_height), (255, 0, 0), -1)
#             frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
#             cv2.rectangle(overlay, (region_start, 0), (region_end, frame_height), (0, 0, 255), -1)
#             frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)


#             current_centroids = []


#             for _, row in detections.iterrows():
#                 cname = row['name'].lower()
#                 if row['confidence'] < CONF_THRESHOLD or cname not in allowed_classes:
#                     continue
#                 xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])


#                 if cname == "box":
#                     cx = (xmin + xmax) // 2
#                     cy = (ymin + ymax) // 2


#                     matched_id = None
#                     for oid, (pcx, pcy) in active_objects.items():
#                         if abs(cx - pcx) < 50 and abs(cy - pcy) < 50:
#                             matched_id = oid
#                             break
#                     if matched_id is None:
#                         matched_id = next_object_id
#                         object_id = next_object_id + 1 if 'object_id' in locals() else 1
#                         next_object_id = object_id


#                     active_objects[matched_id] = (cx, cy)


#                     if matched_id not in box_history:
#                         box_history[matched_id] = deque(maxlen=5)
#                     box_history[matched_id].append((xmin, ymin, xmax, ymax))
#                     box_confidence[matched_id] = row['confidence']


#                     sm_x1, sm_y1, sm_x2, sm_y2 = smooth_box(box_history[matched_id])
#                     if matched_id not in object_states:
#                         object_states[matched_id] = 0
#                     if region_start <= sm_x1 <= region_end:
#                         if object_states[matched_id] == 0:
#                             object_states[matched_id] = 1
#                     else:
#                         if object_states[matched_id] == 1:
#                             count += 1
#                             object_states[matched_id] = 0
#                             class_counts["box"] = class_counts.get("box", 0) + 1


#                     cv2.rectangle(frame, (sm_x1, sm_y1), (sm_x2, sm_y2), (0, 255, 0), 2)
#                     cv2.putText(
#                         frame, f"{cname} {row['confidence']:.2f}",
#                         (sm_x1, sm_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5, (0, 255, 0), 2
#                     )


#                 elif cname in ["4box", "5box", "6box"]:
#                     centroid = get_centroid((xmin, ymin, xmax, ymax))
#                     current_centroids.append({
#                         "centroid": centroid,
#                         "bbox": (xmin, ymin, xmax, ymax),
#                         "class_name": cname,
#                         "confidence": row['confidence']
#                     })


#             updated_trackers = {}
#             used_ids = set()
#             line_horizontal = frame_height // 2  # horizontal center line for 4/5/6 box counting


#             for obj in current_centroids:
#                 cx, cy = obj["centroid"]
#                 bbox = obj["bbox"]
#                 class_name = obj["class_name"]
#                 confidence = obj["confidence"]


#                 assigned_id = None
#                 min_dist = float('inf')


#                 for obj_id, prev_cy in object_trackers.items():
#                     if obj_id in used_ids:
#                         continue
#                     dist = abs(prev_cy - cy)
#                     if dist < 50 and dist < min_dist:
#                         min_dist = dist
#                         assigned_id = obj_id


#                 if assigned_id is None:
#                     assigned_id = next_object_id
#                     next_object_id += 1


#                 if assigned_id in object_trackers:
#                     if is_crossing_line(object_trackers[assigned_id], cy, line_horizontal):
#                         increment = {"4box": 4, "5box": 5, "6box": 6}.get(class_name.lower(), 1)
#                         count += increment
#                         class_counts[class_name] = class_counts.get(class_name, 0) + 1


#                 updated_trackers[assigned_id] = cy
#                 used_ids.add(assigned_id)


#                 xmin, ymin, xmax, ymax = bbox
#                 color = (0, 255, 0)
#                 label = f"{class_name} {confidence:.2f} ID:{assigned_id}"
#                 cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
#                 cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
#                 cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)


#             object_trackers = updated_trackers


#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 2.0
#             thickness = 3
#             text = f"Count: {count}"
#             (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#             x = (frame_width - text_width) // 2
#             y = 50
#             cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 255), thickness)


#             details_text = ", ".join(f"{v} {k}" for k, v in class_counts.items())
#             cv2.putText(frame, details_text, (x, y + 40), font, 1.0, (0, 255, 0), 2)


#             cv2.imshow("Live Detection with Counting", frame)


#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q') or not processing_flag:
#                 logger.info(f"Video processing stopped by 'q' or stop flag, count={count}")
#                 break
        
#         cap.release()
#         cv2.destroyAllWindows()


#         end_time = datetime.datetime.now()
#         logger.info(f"Video processing ended at: {end_time}")
#         elapsed_time = end_time - start_time
#         logger.info(f"Total processing time: {elapsed_time}")
#         logger.info(f"Total boxes counted crossing line: {count}")

#         processing_result = {
#             "count": count,
#             "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
#             "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
#             "processing_duration_sec": elapsed_time.total_seconds(),
#             "model_used": selected_model,
#             "classes_detected": classes
#         }

#         send_count_email(count, "abinayabi55@gmail.com", start_time, end_time, supervisor_name, vehicle_no, class_counts)

#     except Exception as e:
#         logger.error(f"Error in processing video: {e}")
#         processing_flag = False


# @app.post("/process-video")
# async def process_video(
#         file: UploadFile = File(None),
#         video_url: str = Form(None),
#         supervisor_name: str = Form(...),
#         vehicle_no: str = Form(...),
#         selected_model: str = Form(...),
#         background_tasks: BackgroundTasks = BackgroundTasks()
# ):
#     global processing_flag, processing_thread, processing_result
#     if processing_flag:
#         raise HTTPException(status_code=400, detail="Processing already running")

#     processing_flag = True
#     processing_result = {}

#     processing_thread = threading.Thread(target=run_video_processing,
#                                          args=(file, video_url, supervisor_name, vehicle_no, selected_model, background_tasks))
#     processing_thread.start()

#     return {"status": "Processing started"}

# @app.get("/processing-result")
# async def processing_result_api():
#     global processing_flag, processing_result
#     if processing_flag:
#         return JSONResponse({"status": "processing", "result": processing_result})
#     else:
#         return JSONResponse({"status": "done", "result": processing_result})

# @app.post("/stop-processing")
# async def stop_processing():
#     global processing_flag
#     if not processing_flag:
#         return {"status": "Processing not running"}
#     processing_flag = False
#     return {"status": "Processing stopping"}

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, exc: RequestValidationError):
#     return JSONResponse(
#         status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#         content=jsonable_encoder({
#             "detail": exc.errors(),
#             "body": exc.body
#         }),
#     )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




#---------- above code is ru perfectly for video url as input below code run on live stream---------




# import pathlib
# pathlib.PosixPath = pathlib.WindowsPath
# from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.exceptions import RequestValidationError
# from fastapi.encoders import jsonable_encoder
# from fastapi import status
# import threading
# import torch
# import cv2
# import os
# import tempfile
# import requests
# from torchvision import transforms
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import smtplib
# import logging
# import datetime
# import numpy as np
# from collections import deque
# from dotenv import load_dotenv
# import os

# # Load environment variables

# load_dotenv()
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# app = FastAPI()

# origins = [
#     "http://localhost:3000",  # frontedn
#     "http://localhost:3001",  #backend
    
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# YOLOV5_FULLPATH = "D:/Vchanel/Box_detection_web/yolov5"
# MODEL_PATHS = {
#     "Single Box": {
#         "weights": "D:/Vchanel/Box_detection_web/backend/best5.pt",
#         "classes": ["box"]
#     },
#     "Multiple Box": {
#         "weights": "D:/Vchanel/Box_detection_web/backend/best_demo2.pt",
#         "classes": ["box"]
#     },
#     "4_5_6 Box": {
#         "weights": "D:/Vchanel/Box_detection_web/backend/best_demo_allbox.pt",
#         "classes": ["4box", "5box", "6box"]
#     }
# }

# CONF_THRESHOLD = 0.5
# def smooth_box(history):
#     xs1, ys1, xs2, ys2 = zip(*history)
#     return int(np.mean(xs1)), int(np.mean(ys1)), int(np.mean(xs2)), int(np.mean(ys2))
# def get_centroid(box):
#     xmin, ymin, xmax, ymax = box
#     cx = int((xmin + xmax) / 2)
#     cy = int((ymin + ymax) / 2)
#     return cx, cy
# def is_crossing_line(prev_pos, current_pos, line_y):
#     return prev_pos < line_y and current_pos >= line_y
# def send_count_email(count: int, to_email: str, start_time: datetime.datetime, end_time: datetime.datetime,
#                     supervisor_name: str, vehicle_no: str, class_counts: dict):
    
#     logger.info(f"Preparing to send email for count: {count}")
#     sender_email = os.getenv("EMAIL_ADDRESS")  # Your Gmail address
#     sender_password = os.getenv("EMAIL_PASSWORD")  # Your Gmail app password
#     smtp_server = "smtp.gmail.com"
#     smtp_port = 587
#     message = MIMEMultipart()
#     message["From"] = sender_email
#     message["To"] = to_email
#     message["Subject"] = "Box Count"
#     class_counts_str = "\n".join(f"{cls}: {cnt}" for cls, cnt in class_counts.items())
#     body = (f"Supervisor Name: {supervisor_name}\n"
#             f"Vehicle Number: {vehicle_no}\n\n"
#             f"Total count of Boxes : {count}\n\n"
#             f"Counts per class:\n{class_counts_str}\n\n"
#             f"Video processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
#             f"Video processing ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     message.attach(MIMEText(body, "plain"))
#     try:
#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.set_debuglevel(1)
#         server.starttls()
#         server.login(sender_email, sender_password)
#         server.sendmail(sender_email, to_email, message.as_string())
#         server.quit()
#         logger.info("Email sent successfully!")
#     except Exception as e:
#         logger.error(f"Error sending email: {e}")
        
# @app.get("/video/{filename}")
# def get_video(filename: str):
#     video_path = f"{filename}"
#     if not os.path.isfile(video_path):
#         raise HTTPException(status_code=404, detail="Video not found")
#     return FileResponse(video_path, media_type="video/mp4")
# processing_flag = False
# processing_thread = None
# processing_result = {}

# import os
# def run_video_processing(video_url, supervisor_name, vehicle_no, selected_model, background_tasks):
#     global processing_flag, processing_result
#     if selected_model not in MODEL_PATHS:
#         logger.error(f"Unknown model: {selected_model}")
#         processing_flag = False
#         return
#     config = MODEL_PATHS[selected_model]
#     weights_path = config["weights"]
#     classes = config["classes"]
#     count = 0
#     object_trackers = {}
#     next_object_id = 0
#     class_counts = {}
#     box_history = {}
#     box_confidence = {}
#     active_objects = {}
#     object_states = {}  # 0=outside, 1=inside
#     region_width = 40
#     try:
#         # (model loading and other setup...)
#         if video_url:
#             if video_url.startswith("file://") or os.path.exists(video_url):
#                 video_path = video_url.replace("file://", "")
#                 if not os.path.exists(video_path):
#                     logger.error(f"Local file not found: {video_path}")
#                     processing_flag = False
#                     return
#                 cap = cv2.VideoCapture(video_path)  # local file use normal capture
#             elif video_url.startswith("rtsp://"):
#                 # RTSP stream setup with environment variable for transport UDP
#                 os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
#                 cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
#                 if not cap.isOpened():
#                     logger.error("Cannot open RTSP stream")
#                     processing_flag = False
#                     return
#             elif video_url.startswith("http://") or video_url.startswith("https://"):
#                 # download HTTP video file as before
#                 tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 response = requests.get(video_url, stream=True)
#                 response.raise_for_status()
#                 for chunk in response.iter_content(chunk_size=8192):
#                     if chunk:
#                         tmp_file.write(chunk)
#                 tmp_file.close()
#                 cap = cv2.VideoCapture(tmp_file.name)
#             else:
#                 video_path = video_url
#                 cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         line_position = frame_width // 2  # vertical line position
#         start_time = datetime.datetime.now()
#         logger.info(f"Video processing started at: {start_time}")
        
#         allowed_classes = [cls.lower() for cls in classes]
        
#         while True:
#             ret, frame = cap.read()
#             if not ret or not processing_flag:
#                 break
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             # Load model here or use preloaded (make sure the model is loaded outside the loop ideally)
#             model = torch.hub.load(
#                 YOLOV5_FULLPATH,
#                 'custom',
#                 path=weights_path,
#                 source='local',
#                 force_reload=False
#             )
#             model.eval()
            
#             results = model(frame_rgb)
#             detections = results.pandas().xyxy[0]
#             # Draw vertical red line for count zone
#             cv2.line(frame, (line_position, 0), (line_position, frame_height), (0, 0, 255), 2)
#             region_start = line_position - region_width
#             region_end = line_position + region_width
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (region_start, 0), (region_end, frame_height), (255, 0, 0), -1)
#             frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
#             cv2.rectangle(overlay, (region_start, 0), (region_end, frame_height), (0, 0, 255), -1)
#             frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
#             current_centroids = []
#             for _, row in detections.iterrows():
#                 cname = row['name'].lower()
#                 if row['confidence'] < CONF_THRESHOLD or cname not in allowed_classes:
#                     continue
#                 xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
#                 if cname == "box":
#                     cx = (xmin + xmax) // 2
#                     cy = (ymin + ymax) // 2
#                     matched_id = None
#                     for oid, (pcx, pcy) in active_objects.items():
#                         if abs(cx - pcx) < 50 and abs(cy - pcy) < 50:
#                             matched_id = oid
#                             break
#                     if matched_id is None:
#                         matched_id = next_object_id
#                         object_id = next_object_id + 1 if 'object_id' in locals() else 1
#                         next_object_id = object_id
#                     active_objects[matched_id] = (cx, cy)
#                     if matched_id not in box_history:
#                         box_history[matched_id] = deque(maxlen=5)
#                     box_history[matched_id].append((xmin, ymin, xmax, ymax))
#                     box_confidence[matched_id] = row['confidence']
#                     sm_x1, sm_y1, sm_x2, sm_y2 = smooth_box(box_history[matched_id])
#                     if matched_id not in object_states:
#                         object_states[matched_id] = 0
#                     if region_start <= sm_x1 <= region_end:
#                         if object_states[matched_id] == 0:
#                             object_states[matched_id] = 1
#                     else:
#                         if object_states[matched_id] == 1:
#                             count += 1
#                             object_states[matched_id] = 0
#                             class_counts["box"] = class_counts.get("box", 0) + 1
#                     cv2.rectangle(frame, (sm_x1, sm_y1), (sm_x2, sm_y2), (0, 255, 0), 2)
#                     cv2.putText(
#                         frame, f"{cname} {row['confidence']:.2f}",
#                         (sm_x1, sm_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5, (0, 255, 0), 2
#                     )
#                 elif cname in ["4box", "5box", "6box"]:
#                     centroid = get_centroid((xmin, ymin, xmax, ymax))
#                     current_centroids.append({
#                         "centroid": centroid,
#                         "bbox": (xmin, ymin, xmax, ymax),
#                         "class_name": cname,
#                         "confidence": row['confidence']
#                     })
#             updated_trackers = {}
#             used_ids = set()
#             line_horizontal = frame_height // 2  # horizontal center line for 4/5/6 box counting
#             for obj in current_centroids:
#                 cx, cy = obj["centroid"]
#                 bbox = obj["bbox"]
#                 class_name = obj["class_name"]
#                 confidence = obj["confidence"]
#                 assigned_id = None
#                 min_dist = float('inf')
#                 for obj_id, prev_cy in object_trackers.items():
#                     if obj_id in used_ids:
#                         continue
#                     dist = abs(prev_cy - cy)
#                     if dist < 50 and dist < min_dist:
#                         min_dist = dist
#                         assigned_id = obj_id
#                 if assigned_id is None:
#                     assigned_id = next_object_id
#                     next_object_id += 1
#                 if assigned_id in object_trackers:
#                     if is_crossing_line(object_trackers[assigned_id], cy, line_horizontal):
#                         increment = {"4box": 4, "5box": 5, "6box": 6}.get(class_name.lower(), 1)
#                         count += increment
#                         class_counts[class_name] = class_counts.get(class_name, 0) + 1
#                 updated_trackers[assigned_id] = cy
#                 used_ids.add(assigned_id)
#                 xmin, ymin, xmax, ymax = bbox
#                 color = (0, 255, 0)
#                 label = f"{class_name} {confidence:.2f} ID:{assigned_id}"
#                 cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
#                 cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
#                 cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
#             object_trackers = updated_trackers
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 2.0
#             thickness = 3
#             text = f"Count: {count}"
#             (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#             x = (frame_width - text_width) // 2
#             y = 50
#             cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 255), thickness)
#             details_text = ", ".join(f"{v} {k}" for k, v in class_counts.items())
#             cv2.putText(frame, details_text, (x, y + 40), font, 1.0, (0, 255, 0), 2)
#             cv2.imshow("Live Detection with Counting", frame)
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q') or not processing_flag:
#                 logger.info(f"Video processing stopped by 'q' or stop flag, count={count}")
#                 break
#         cap.release()
#         cv2.destroyAllWindows()
#         end_time = datetime.datetime.now()
#         logger.info(f"Video processing ended at: {end_time}")
#         elapsed_time = end_time - start_time
#         logger.info(f"Total processing time: {elapsed_time}")
#         logger.info(f"Total boxes counted crossing line: {count}")
#         processing_result = {
#             "count": count,
#             "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
#             "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
#             "processing_duration_sec": elapsed_time.total_seconds(),
#             "model_used": selected_model,
#             "classes_detected": classes
#         }
#         send_count_email(count, "abinayabi55@gmail.com", start_time, end_time, supervisor_name, vehicle_no, class_counts)
#     except Exception as e:
#         logger.error(f"Error in processing video: {e}")
#         processing_flag = False
# @app.post("/process-video")
# async def process_video(
#         file: UploadFile = File(None),
#         video_url: str = Form(None),
#         supervisor_name: str = Form(...),
#         vehicle_no: str = Form(...),
#         selected_model: str = Form(...),
#         background_tasks: BackgroundTasks = BackgroundTasks()
# ):
#     global processing_flag, processing_thread, processing_result
#     if processing_flag:
#         raise HTTPException(status_code=400, detail="Processing already running")
#     processing_flag = True
#     processing_result = {}
#     processing_thread = threading.Thread(target=run_video_processing,
#                                          args=(video_url, supervisor_name, vehicle_no, selected_model, background_tasks))
#     processing_thread.start()
#     return {"status": "Processing started"}
# @app.get("/processing-result")
# async def processing_result_api():
#     global processing_flag, processing_result
#     if processing_flag:
#         return JSONResponse({"status": "processing", "result": processing_result})
#     else:
#         return JSONResponse({"status": "done", "result": processing_result})
# @app.post("/stop-processing")
# async def stop_processing():
#     global processing_flag
#     if not processing_flag:
#         return {"status": "Processing not running"}
#     processing_flag = False
#     return {"status": "Processing stopping"}
# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, exc: RequestValidationError):
#     return JSONResponse(
#         status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#         content=jsonable_encoder({
#             "detail": exc.errors(),
#             "body": exc.body
#         }),
#     )
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



# ------ above code is use two counting machanisum i want to use one --------






import pathlib
pathlib.PosixPath = pathlib.WindowsPath

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
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from collections import deque






load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 👇 Add this line to serve your uploaded_videos folder
#app.mount("/videos", StaticFiles(directory="uploaded_videos"), name="videos")

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
YOLOV5_FULLPATH = os.path.abspath("yolov5")

MODEL_PATHS = {
    "Single Box": {
        "weights":  os.path.abspath("backend/best5.pt"),
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
    """
    Smooths bounding box coordinates by averaging the history.
    Args:
        history (deque): A deque containing historical bounding box coordinates.
    Returns:
        tuple: The smoothed bounding box coordinates (xmin, ymin, xmax, ymax).
    """
    xs1, ys1, xs2, ys2 = zip(*history)
    return int(np.mean(xs1)), int(np.mean(ys1)), int(np.mean(xs2)), int(np.mean(ys2))
def get_centroid(box):
    """
    Calculates the centroid of a bounding box.
    Args:
        box (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
    Returns:
        tuple: Centroid coordinates (cx, cy).
    """
    xmin, ymin, xmax, ymax = box
    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)
    return cx, cy
def is_crossing_line(prev_pos, current_pos, line_x):
    """
    Checks if an object's centroid has crossed a vertical line.
    Args:
        prev_pos (int): The previous x-position of the object's centroid.
        current_pos (int): The current x-position of the object's centroid.
        line_x (int): The x-coordinate of the vertical line.
    Returns:
        bool: True if the object has crossed the line, False otherwise.
    """
    return (prev_pos < line_x and current_pos >= line_x) or \
           (prev_pos > line_x and current_pos <= line_x)
def send_count_email(count: int, to_email: str, start_time: datetime.datetime, end_time: datetime.datetime,
                    supervisor_name: str, vehicle_no: str, class_counts: dict):
    """
    Sends an email with the video processing results.
    """
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
    """
    Loads and caches the YOLOv5 model.
    """
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
def run_video_processing(video_url, supervisor_name, vehicle_no, selected_model, background_tasks):
    """
    Main function to process video, detect and count objects.
    """
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
    object_trackers = {} # Stores last known centroid position for each tracked object
    next_object_id = 0
    class_counts = {}
    box_history = {} # For smoothing bounding boxes
    try:
        if video_url:
            if video_url.startswith("file://") or os.path.exists(video_url):
                video_path = video_url.replace("file://", "")
                if not os.path.exists(video_path):
                    logger.error(f"Local file not found: {video_path}")
                    processing_flag = False
                    return
                cap = cv2.VideoCapture(video_path)
            elif video_url.startswith("rtsp://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
                cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logger.error("Cannot open RTSP stream")
                    processing_flag = False
                    return
            elif video_url.startswith("http://") or video_url.startswith("https://"):
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                response = requests.get(video_url, stream=True)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                tmp_file.close()
                cap = cv2.VideoCapture(tmp_file.name)
            else:
                video_path = video_url
                cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_position = frame_width // 2 # Vertical line position for counting
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
                # Count only when the centroid crosses the line
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
            cv2.imshow("Live Detection with Counting", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or not processing_flag:
                logger.info(f"Video processing stopped by 'q' or stop flag, count={count}")
                break
        cap.release()
        cv2.destroyAllWindows()
        end_time = datetime.datetime.now()
        logger.info(f"Video processing ended at: {end_time}")
        elapsed_time = end_time - start_time
        logger.info(f"Total processing time: {elapsed_time}")
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
    processing_thread = threading.Thread(target=run_video_processing,
                                         args=(video_url, supervisor_name, vehicle_no, selected_model, background_tasks))
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
