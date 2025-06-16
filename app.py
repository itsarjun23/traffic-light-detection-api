from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO


app = FastAPI(title="Traffic Light Detection API")

try:
    model = YOLO("traffic_lights_yolov11.pt")  
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

@app.get("/")
async def home():
    return {"message": "Welcome to Traffic Light Detection API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        
        results = model(image)

        
        if not results or results[0].boxes is None:
            return {"predictions": []}

        
        predictions = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            predictions.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name
            })

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
