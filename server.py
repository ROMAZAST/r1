from flask import Flask, request, jsonify
from datetime import datetime
import cv2
import numpy as np
import pytesseract
import re
import time
import concurrent.futures
import json
from PIL import Image
import threading
import queue
app = Flask(__name__)


last_logged_plate = None
last_logged_time = 0

DB_PATH = "db.json"
def load_db():
    try:
        with open(DB_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"plates": {}}

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=4)

from datetime import datetime, timedelta

def update_plate_in_db(plate, min_interval_sec=10):
    db = load_db()
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    plate_data = db["plates"].get(plate, {"count": 0, "last_seen": "2000-01-01 00:00:00"})

    try:
        last_seen_time = datetime.strptime(plate_data["last_seen"], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        last_seen_time = datetime(2000, 1, 1)

    if (now - last_seen_time).total_seconds() < min_interval_sec:
        log_event(f"⏱ {plate} was updated recently — skipped DB update")
        return

    plate_data["count"] += 1
    plate_data["last_seen"] = now_str
    db["plates"][plate] = plate_data
    save_db(db)
    log_event(f" Updated {plate} in DB")
def crop_right_of_yellow(img, yellow_thresh=0.001):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 40, 40]), np.array([40, 255, 255]))
    h, w = mask.shape
    yellow_columns = np.sum(mask > 0, axis=0)
    for x in range(w - 1, -1, -1):
        if yellow_columns[x] / h >= yellow_thresh:
            return img[:, x + 1:]
    return img


def log_event(message):
    print(f"[LOG] {message}")




def process_angle(image, angle, config, pattern):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    cropped = crop_right_of_yellow(rotated)
    if cropped is None or cropped.size == 0:
        return []

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    processed = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)

    text = pytesseract.image_to_string(processed, config=config)
    cleaned = re.sub(r"[^A-ZА-Я0-9]", "", text.upper())
    return pattern.findall(cleaned)

def recognize_top_plates(image: np.ndarray):
    global last_logged_plate, last_logged_time

    start_time = time.time()
    if image is None or image.size == 0:
        return []

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯ0123456789"
    pattern = re.compile(r"[A-ZА-Я]{2}\d{4}[A-ZА-Я]{2}")

    angles = np.arange(-3, 3, 0.1)
    hits = {}
    lock = threading.Lock()
    stop_event = threading.Event()
    def worker(angle):
        if stop_event.is_set():
            return []
        matches = process_angle(image, angle, config, pattern)
        with lock:
            for match in matches:
                hits[match] = hits.get(match, 0) + 1
                if hits[match] >= 7:
                    stop_event.set()
        return matches

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(angles)) as executor:
        futures = [executor.submit(worker, angle) for angle in angles]

        concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED if stop_event.is_set() else concurrent.futures.ALL_COMPLETED)

    print(f"🧠 TOTAL OCR thread time (parallel): {(time.time() - start_time) * 1000:.1f} ms")
    if hits:
        top2 = sorted(hits.items(), key=lambda x: -x[1])[:2]
        top_counts = [count for _, count in top2]
        total = sum(top_counts) if top_counts else 1
        now = time.time()
        results = []

        global last_logged_plate, last_logged_time  # <--- ДОДАЙ ЦЕ

        for plate, count in top2:
            percent = (count / total) * 100 if total > 0 else 0
            #print(f"Plate: {plate}, Count: {count}, Confidence: {percent:.2f}%")
            if percent >= 65 :
                log_event(f"{plate} was detected, percent: {percent:.2f}%")
                last_logged_plate = plate
                last_logged_time = now
                if count >= 4:
                    update_plate_in_db(plate)

            results.append({"plate": plate, "confidence": percent})
        #print(results)
        return results

    return None


def queue_worker():
    while True:
        request_id, image_cv = task_queue.get()
        result = recognize_top_plates(image_cv)
        results_map[request_id] = result
        task_queue.task_done()

task_queue = queue.Queue()
request_id_lock = threading.Lock()
results_map = {}
request_id_counter = 0

threading.Thread(target=queue_worker, daemon=True).start()

@app.route("/upload", methods=["POST"])
def upload_image():
    global request_id_counter
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


    with request_id_lock:
        request_id = request_id_counter
        request_id_counter += 1


    task_queue.put((request_id, image_cv))


    while request_id not in results_map:
        time.sleep(0.01)

    result = results_map.pop(request_id)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)