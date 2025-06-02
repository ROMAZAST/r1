import cv2
import time
from ultralytics import YOLO

def run_detection_test(cap_width, cap_height, num_tests, model):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    detection_times = []
    print(f"\nRunning {num_tests} detections at resolution {cap_width}x{cap_height}...")

    while len(detection_times) < num_tests:
        ret, frame = cap.read()
        if not ret:
            continue

        start_time = time.time()
        _ = model(frame, verbose=False)
        elapsed = time.time() - start_time
        detection_times.append(elapsed)
        print(f"[{len(detection_times)}] Time: {elapsed:.3f} sec")

    cap.release()

    avg_time = sum(detection_times) / len(detection_times)
    min_time = min(detection_times)
    max_time = max(detection_times)

    print(f"\n=== Stats for {cap_width}x{cap_height} ===")
    print(f"Total detections: {num_tests}")
    print(f"Average time: {avg_time:.3f} sec")
    print(f"Minimum time: {min_time:.3f} sec")
    print(f"Maximum time: {max_time:.3f} sec")
    print("=" * 40)

# Load the YOLO model
model = YOLO("best_2.pt")

# Run tests
run_detection_test(1920, 1080, 1000, model)
run_detection_test(640, 480, 100, model)
