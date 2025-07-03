import cv2
import os
from models.yolov12 import YOLOv12

# Inisialisasi model sekali saja agar tidak loading ulang setiap tracking
model = YOLOv12("weights/yolo12n.pt")

def run_tracking(source, output):
    cap = cv2.VideoCapture(source)

    # Ambil informasi resolusi dan fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Fallback jika tidak terbaca
    if width == 0 or height == 0:
        print("[WARNING] Resolusi tidak terbaca, fallback ke 640x480")
        width, height = 640, 480
    if fps == 0 or fps is None:
        print("[WARNING] FPS tidak terbaca, fallback ke 20")
        fps = 20

    print(f"[INFO] Output to: {output}")
    print(f"[INFO] Frame size: {width}x{height}, FPS: {fps}")

    # Codec dan VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # bisa diganti 'XVID' untuk .avi
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    if not out.isOpened():
        print("[ERROR] Gagal membuka VideoWriter!")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model.track(frame)  # Lakukan tracking per frame
        out.write(result)
        frame_count += 1

    cap.release()
    out.release()
    print(f"[INFO] Tracking selesai, {frame_count} frame diproses.")
