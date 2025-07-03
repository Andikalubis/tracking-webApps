import os
import time
import cv2
from models.yolov12 import YOLOv12

def run_detection(source):
    model = YOLOv12("weights/yolo12n.pt")

    img = cv2.imread(source)
    result = model.track(img)

    filename = os.path.basename(source).rsplit('.', 1)[0]
    detected_filename = f"{filename}_{int(time.time())}_detected.jpg"
    output_path = os.path.join("static/uploads", detected_filename)

    cv2.imwrite(output_path, result)
    return detected_filename
