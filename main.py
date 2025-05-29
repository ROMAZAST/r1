from ultralytics import YOLO
from datetime import datetime
import cv2
import numpy as np
import pytesseract
import re
import time
import threading
import json


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

def update_plate_in_db(plate):
    db = load_db()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plate_data = db["plates"].get(plate, {"count": 0})
    plate_data["count"] += 1
    plate_data["last_seen"] = now_str
    db["plates"][plate] = plate_data
    save_db(db)
    print(f"✅ Updated {plate} in DB")
def crop_right_of_yellow(img, yellow_thresh=0.001):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 40, 40]), np.array([40, 255, 255]))
    h, w = mask.shape
    yellow_columns = np.sum(mask > 0, axis=0)
    for x in range(w - 1, -1, -1):
        if yellow_columns[x] / h >= yellow_thresh:
            return img[:, x + 1:]
    return img



def log_event(msg: str):
    global log_lines
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_lines.append(f"[{timestamp}] {msg}")
    if len(log_lines) > 10:
        log_lines = log_lines[-10:]


def show_log_window():
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    y = 20
    for line in log_lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 18
    cv2.imshow("Event Log", img)

def clean(text):
    return re.sub(r"[^A-ZА-Я0-9]", "", text.upper())
import concurrent.futures

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
    start_time = time.time()

    if image is None or image.size == 0:
        return []

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯ0123456789"
    pattern = re.compile(r"[A-ZА-Я]{2}\d{4}[A-ZА-Я]{2}")

    angles = np.arange(-3, 3, 0.5)
    hits = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(angles)) as executor:
        futures = [executor.submit(process_angle, image, angle, config, pattern) for angle in angles]
        for future in concurrent.futures.as_completed(futures):
            try:
                matches = future.result()
                for match in matches:
                    hits[match] = hits.get(match, 0) + 1
            except Exception as e:
                print(f"Thread error: {e}")

    print(f"🧠 TOTAL OCR thread time (parallel): {(time.time() - start_time) * 1000:.1f} ms")
    top2 = sorted(hits.items(), key=lambda x: -x[1])[:2]
    if top2:
        plate, count = top2[0]
        total = sum(hits.values())
        percent = (count / total) * 100 if total > 0 else 0
        global last_logged_plate, last_logged_time
        now = time.time()
        if percent >= 65 and (plate != last_logged_plate or now - last_logged_time >= 5):
            log_event(f"{plate} was recognized by camera, percent: {percent:.2f}%")
            last_logged_plate = plate
            last_logged_time = now
            if count >=4:
                update_plate_in_db(plate)
            return None
        return None
    return None


model = YOLO("best_downloaded.pt")
cap = cv2.VideoCapture(0)
log_lines = []
last_logged_plate = ""
last_logged_time = 0

ocr_roi = None
ocr_result_text = ""
last_ocr_time = 0
ocr_interval = 0.5
ocr_busy = False
roi_for_ocr = None
ocr_lock = threading.Lock()
ocr_start_time = 0

def ocr_thread_func():
    global roi_for_ocr, ocr_result_text
    while True:
        with ocr_lock:
            roi = roi_for_ocr.copy() if roi_for_ocr is not None else None
            roi_for_ocr = None

        if roi is not None:
            try:
                result_with_plates = recognize_top_plates(roi)
                if result_with_plates:
                    text = "\n".join([f"{p}: {c} times" for p, c in result_with_plates])
                    with ocr_lock:
                        ocr_result_text = text
            except Exception as e:
                print(f"OCR error: {e}")

        time.sleep(0.1)

ocr_thread = threading.Thread(target=ocr_thread_func, daemon=True)
ocr_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        h, w = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if current_time - last_ocr_time >= ocr_interval:
                with ocr_lock:
                    roi_for_ocr = roi.copy()
                    ocr_start_time = time.time()
                last_ocr_time = current_time


    #cv2.imshow("Cam[1]", frame)


    show_log_window()
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
