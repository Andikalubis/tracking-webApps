from ultralytics import YOLO

video_path = 'input.mp4'  
model_path = 'weights/yolo12n.pt'

print("[INFO] Memuat model dengan tracking...")
model = YOLO(model_path)

print("[INFO] Mulai tracking...")
results = model.track(source=video_path, save=True, tracker="botsort.yaml")

print("[SELESAI] Video dengan tracking disimpan di folder 'runs/track'")
