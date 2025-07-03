from ultralytics import YOLO
import cv2

class YOLOv12:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image):
        return self.model(image)[0]

    def track(self, frame):
        # Melakukan inferensi pada 1 frame dan mengembalikan frame dengan bounding box
        results = self.model.track(frame, persist=True, conf=0.4)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame
