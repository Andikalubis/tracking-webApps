from flask import Flask, render_template, request, redirect, Response
import os
import cv2
import time
from werkzeug.utils import secure_filename
from video_track import run_tracking
from video_detect import run_detection
from models.yolov12 import YOLOv12

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLOv12("weights/yolo12n.pt")
camera = None  # webcam global agar hanya dibuka jika dipilih

@app.route('/')
def index():
    return render_template('index.html', selected_mode='webcam')

@app.route('/detect', methods=['POST'])
def detect():
    mode = request.form['mode']
    file = request.files.get('file')

    filepath = None
    if file and file.filename != "":
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

    if mode == 'webcam':
        return redirect('/?mode=webcam')
    elif mode == 'video' and filepath:
        filename = f"{os.path.basename(filepath).rsplit('.', 1)[0]}_{int(time.time())}_tracked.mp4"
        output_path = os.path.join(UPLOAD_FOLDER, filename)
        run_tracking(source=filepath, output=output_path)
        return redirect(f"/result?video={filename}&mode=video")

    elif mode == 'image' and filepath:
        detected_filename = run_detection(source=filepath)
        return redirect(f"/result?image={detected_filename}&mode=image")
    else:
        return "❌ File dibutuhkan untuk mode ini", 400

def generate_stream():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        result = model.track(frame)
        ret, buffer = cv2.imencode('.jpg', result)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if request.args.get("mode") == "webcam":
        return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Forbidden", 403

@app.route('/result')
def result():
    image = request.args.get('image')
    video = request.args.get('video')
    mode = request.args.get('mode', '')

    image_url = f"/static/uploads/{image}" if image else None
    video_url = f"/static/uploads/{video}" if video else None

    return render_template("index.html", image_url=image_url, video_url=video_url, selected_mode=mode)

if __name__ == '__main__':
    app.run(debug=True)
