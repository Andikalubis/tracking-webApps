import cv2
from ultralytics import YOLO

def run_video(source):
    model = YOLO("weights/yolo12n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Gagal membuka sumber video.")
        return

    print("🎥 Tracking dimulai... (tekan 'q' untuk keluar)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi objek
        results = model.predict(frame, conf=0.3, verbose=False)

        # Ambil prediksi dan gambar kotak deteksi
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = box.conf[0]
                label = model.names[cls]
                text = f"{label} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv12 Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n🧠 Pilih mode:")
    print("1. Tracking dari Webcam")
    print("2. Tracking dari File Video")
    mode = input("Masukkan nomor mode [1/2]: ").strip()

    if mode == "1":
        run_video(0)
    elif mode == "2":
        path = input("Masukkan path ke file video: ").strip()
        run_video(path)
    else:
        print("❌ Mode tidak valid.")
