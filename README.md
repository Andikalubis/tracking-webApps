🎯 Real-Time Object Tracking WebApp with YOLOv12
Proyek ini adalah implementasi YOLOv12 dengan Ultralytics API untuk object detection & tracking secara real-time menggunakan webcam maupun video file. Sistem mendeteksi dan melacak objek secara langsung dengan akurasi tinggi dan performa optimal di lingkungan desktop lokal.

🧠 Fitur Utama
🔍 Object Tracking Realtime via webcam (cv2.VideoCapture(0))

📹 Tracking dari File Video (input.mp4)

🧠 Menggunakan model YOLOv12n ringan dan cepat

✅ Integrasi Ultralytics API resmi (tanpa custom model/torch.load manual)


🛠️ Spesifikasi & Teknologi
- Komponen	            => Rincian
- Framework	            => Ultralytics YOLO
- Bahasa pemrograman    => Python 3.10
- Tracking Engine	    => BYTETracker via ultralytics.trackers
- Model	                => weights/yolo12n.pt (custom lightweight YOLOv12)
- Video Input	        => Webcam / File Video (.mp4)
- Visualization	        => OpenCV
- Virtual Env	        => venv/ Python virtual environment
- Dependency Mgmt	    => requirements.txt


✅ Cara Menjalankan

Aktifkan virtual environment:
python -m venv venv

Install dependencies:
pip install -r requirements.txt

Jalankan aplikasi:
python app.py

Pilih mode:
🧠 Pilih mode:
1. Tracking dari Webcam
2. Tracking dari File Video
Masukkan nomor mode [1/2]:

⚠️ Catatan
Pastikan webcam aktif untuk mode 1.

Gantilah file video pada input.mp4 sesuai kebutuhan.