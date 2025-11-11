from pathlib import Path
from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("nikimauzer-rscbx").project("cs2-shsmv")
version = project.version(1)
dataset = version.download("yolov12")

data_yaml = Path(dataset.location) / "data.yaml"
model = YOLO("yolov12s.yaml")
model.train(
    data=str(data_yaml),
    epochs=20,
    imgsz=1024,
    patience=50,
    save=True,
    plots=True,
)
