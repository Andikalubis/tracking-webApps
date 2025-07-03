from ultralytics import YOLO

video_path = 'input.mp4'      # Ganti dengan path videomu
model_path = 'weights/yolo12n.pt'

print("[INFO] Memuat model...")
model = YOLO(model_path)

print("[INFO] Mulai deteksi video...")
results = model.predict(source=video_path, save=True)

print("[SELESAI] Video dengan bounding box disimpan di folder 'runs/detect'")
