import cv2
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load emoji with alpha channel
emoji = cv2.imread("emoji.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        emoji_resized = cv2.resize(emoji, (w, h))

        for i in range(h):
            for j in range(w):
                if emoji_resized[i, j][3] > 0:  # alpha channel
                    frame[y + i, x + j] = emoji_resized[i, j][:3]

    cv2.imshow("DAY 8 - Emoji Face Filter 😄", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
