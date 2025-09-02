from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import cv2
import requests
import tempfile
import os

app = FastAPI()

# Configuration - update paths as needed
YOLOV5_FULLPATH = "D:\Vchanel\Box_detection_web\yolov5"
WEIGHTS_PATH = "best_demo_allbox.pt"
CONF_THRESHOLD = 0.8

print("Loading model...")
model = torch.hub.load(
    YOLOV5_FULLPATH,
    'custom',
    path=WEIGHTS_PATH,
    source='local',
    force_reload=False
)
model.eval()
print("Model loaded!")

class VideoURL(BaseModel):
    url: str

@app.post("/process-video")
async def process_video(video: VideoURL):
    # Download the video to temp file
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        response = requests.get(video.url, stream=True)
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)
        tmp_file.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

    cap = cv2.VideoCapture(tmp_file.name)
    if not cap.isOpened():
        os.remove(tmp_file.name)
        raise HTTPException(status_code=400, detail="Could not open video file")

    results_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        detections = results.pandas().xyxy[0]

        frame_detections = []
        for _, row in detections.iterrows():
            if row['confidence'] >= CONF_THRESHOLD:
                frame_detections.append({
                    'class': row['name'],
                    'confidence': float(row['confidence']),
                    'bbox': [
                        float(row['xmin']),
                        float(row['ymin']),
                        float(row['xmax']),
                        float(row['ymax'])
                    ]
                })

        results_list.append(frame_detections)

    cap.release()
    os.remove(tmp_file.name)

    return {"results": results_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video_processing_api:app", host="0.0.0.0", port=8000, reload=True)
