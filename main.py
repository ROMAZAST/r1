import cv2

# Відкриваємо камеру з високим FPS (якщо підтримується)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Для Linux, наприклад Raspberry Pi

# Налаштовуємо роздільну здатність (менша = менше затримка)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Можна зменшити до 25 або 15

# Виключаємо кешування кадрів (важливо!)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Вікно без затримок
cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Зчитуємо лише найсвіжіший кадр (читаємо 1 раз)
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Live", frame)

    # Використовуємо waitKey з малим значенням (1 мс)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC = вихід
        break

cap.release()
cv2.destroyAllWindows()
