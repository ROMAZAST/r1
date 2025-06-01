import cv2
import numpy as np
import requests
import threading
import time
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response

# Глобальні змінні
log_lines = []
search_query = ""
last_sent_time = 0
send_interval = 2.0
last_plate_log_times = {}
output_frame = None
lock = threading.Lock()

model = YOLO("best_2.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
app = Flask(__name__)

def log_event(msg: str):
    global log_lines
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_lines.append(f"[{timestamp}] {msg}")
    if len(log_lines) > 10:
        log_lines = log_lines[-10:]

def show_log_window():
    global search_query
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    y = 20
    filtered_lines = [line for line in log_lines if search_query.lower() in line.lower()] if search_query else log_lines
    for line in filtered_lines[-10:]:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 18
    cv2.putText(img, f"Search: {search_query}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.imshow("Event Log", img)

def send_image_to_server(image, server_url="http://192.168.0.104:5000/upload"):
    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'image': ('plate.jpg', img_encoded.tobytes(), 'image/jpeg')}
        response = requests.post(server_url, files=files)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                for item in result:
                    plate = item.get("plate", "None")
                    conf = item.get("confidence", 0)
                    if conf >= 50:
                        now = time.time()
                        last_time = last_plate_log_times.get(plate, 0)
                        if now - last_time >= 10:
                            log_event(f"Plate {plate} was detected, confidence: {conf:.2f}%")
                            last_plate_log_times[plate] = now
        else:
            log_event(f"❌ Server error {response.status_code}")
    except Exception as e:
        log_event(f"❌ Error sending image: {e}")

def start_upload_thread(plate_image):
    thread = threading.Thread(target=send_image_to_server, args=(plate_image,))
    thread.start()

def main_loop():
    global search_query, last_sent_time, output_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        results = model(frame, verbose=False)
        h, w = frame.shape[:2]
        found_plate = False

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = frame[y1:y2, x1:x2]
                found_plate = True

                if current_time - last_sent_time >= send_interval:
                    start_upload_thread(roi.copy())
                    last_sent_time = current_time
                    break
            if found_plate:
                break

        with lock:
            output_frame = frame.copy()

        show_log_window()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 8:
            search_query = search_query[:-1]
        elif key != 255 and 32 <= key <= 126:
            search_query += chr(key)

    cap.release()
    cv2.destroyAllWindows()

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=main_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=8080)
