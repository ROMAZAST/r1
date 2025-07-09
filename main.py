import cv2

cap = cv2.VideoCapture(0)  # 0 - USB-камера, 1 - Raspberry Pi Camera (залежно від підключення)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))  # Задаємо потрібну якість
    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
