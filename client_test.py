import cv2
import numpy as np
import requests
import threading
import time
from datetime import datetime
from ultralytics import YOLO

# Глобальні змінні
log_lines = []
search_query = ""
last_sent_time = 0
send_interval = 3.0  # секунди
last_plate_log_times = {}

model = YOLO("best_2.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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
#server_url

def send_image_to_server(image, server_url = "http://localhost:5000/upload"):
    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'image': ('plate.jpg', img_encoded.tobytes(), 'image/jpeg')}
        response = requests.post(server_url, files=files)
        # if response.status_code == 200:
        #     result = response.json()
        #     if isinstance(result, list):
        #         for item in result:
        #             plate = item.get("plate", "None")
        #             conf = item.get("confidence", 0)
        #             if conf >= 51:
        #                 now = time.time()
        #                 last_time = last_plate_log_times.get(plate, 0)
        #                 if now - last_time >= 10:
        #                     log_event(f"Plate {plate} detected, confidence: {conf:.2f}%")
        #                     last_plate_log_times[plate] = now
        # else:
        #     log_event(f"❌ Server error {response.status_code}: {response.text}")
    except Exception as e:
        log_event(f"❌ Error sending image: {e}")

def start_upload_thread(plate_image):
    thread = threading.Thread(target=send_image_to_server, args=(plate_image,))
    thread.start()
detection_times = []
def main():
    global search_query, last_sent_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        global detection_times
        max_measurements = 1000

        #results = model(frame, verbose=False)
        start_detect = time.time()
        results = model(frame, verbose=False)
        elapsed_detect = time.time() - start_detect


        if len(detection_times) < max_measurements:
            detection_times.append(elapsed_detect)
            log_event(f"Recognition time #{len(detection_times)}: {elapsed_detect:.3f} sec")
        elif len(detection_times) == max_measurements:
            avg_time = sum(detection_times) / len(detection_times)
            min_time = min(detection_times)
            max_time = max(detection_times)
            log_event(f"Detections count: {len(detection_times)}")
            log_event(f"Average detection time: {avg_time:.3f} sec")
            log_event(f"Minimum detection time: {min_time:.3f} sec")
            log_event(f"Maximum detection time: {max_time:.3f} sec")
            detection_times.append(-1)  

        h, w = frame.shape[:2]
        found_plate = False

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                roi = frame[y1:y2, x1:x2]
                found_plate = True

                if current_time - last_sent_time >= send_interval:
                    start_upload_thread(roi.copy())
                    last_sent_time = current_time
                    break

            if found_plate:
                break

        show_log_window()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 8:  # Backspace
            search_query = search_query[:-1]
        elif key != 255 and 32 <= key <= 126:
            search_query += chr(key)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

