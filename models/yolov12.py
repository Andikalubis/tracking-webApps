from ultralytics import YOLO
from PIL import Image

class YOLOv12:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def detect(self, image_path):
        results = self.model(image_path)
        images = []
        for result in results:
            img = result.plot()
            images.append(Image.fromarray(img))
        return images