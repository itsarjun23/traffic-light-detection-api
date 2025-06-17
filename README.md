# traffic-light-detection-api

This project is a solution for the Object Detection MVP using the **YOLOv11** model and **FastAPI** framework. It includes training on a custom traffic light dataset and deploying a REST API that accepts image input and returns predictions.

---


A. Setup Instructions:

Follow these steps to set up the project and run the API locally:

1. Clone the Repository

You can either clone this repo or download it as a ZIP.

2. Install Python Dependencies

"pip install -r requirements.txt"

3. Run the FastAPI Server

"uvicorn app:app --reload"

---


B. Model Details and Performance Results:

| Metric               | Value         |
|----------------------|---------------|
| Model Used           | YOLOv11 (Ultralytics) |
| Training Dataset     | Custom Traffic Light Dataset |
| Training Platform    | Google Colab (GPU) |
| Total Images         | 487 |
| Total Instances      | 487 |
| Precision (overall)  | 0.999 |
| Recall (overall)     | 1.000 |
| mAP@0.5              | 0.995 |
| mAP@0.5:0.95         | 0.783 |

 Class-wise Metrics

| Class   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---------|-----------|--------|---------|--------------|
| Red     | 1.000     | 1.000  | 0.995   | 0.787        |
| Yellow  | 0.998     | 1.000  | 0.995   | 0.788        |
| Green   | 0.998     | 1.000  | 0.995   | 0.774        |

---


C. Instructions to Run the API Project:

1. Running the API

uvicorn app:app --reload

2. Testing the API via curl

curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@test.jpg"

3. Expected Response Format

{"predictions":[{"x1":143.21035766601562,"y1":0.1862640380859375,"x2":297.6431884765625,"y2":177.98976135253906,"confidence":0.8129075765609741,"class_id":0,"class_name":"red"}]}




