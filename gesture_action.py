import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

finger_tips = [8, 12, 16, 20]

paused = False
last_action_time = 0
cooldown = 1.0  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "No Hand"

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            lm = hand.landmark
            fingers = []

            for tip in finger_tips:
                if lm[tip].y < lm[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Detect gesture
            if fingers == [0, 0, 0, 0]:
                gesture = "FIST ✊"
            elif fingers == [1, 1, 1, 1]:
                gesture = "OPEN PALM ✋"
            elif fingers == [1, 1, 0, 0]:
                gesture = "V SIGN ✌️"
            else:
                gesture = "UNKNOWN"

            # Cooldown logic
            current_time = time.time()
            if current_time - last_action_time > cooldown:
                if gesture == "FIST ✊":
                    paused = True
                    last_action_time = current_time

                elif gesture == "OPEN PALM ✋":
                    paused = False
                    last_action_time = current_time

        status_text = "PAUSED ⏸️" if paused else "RUNNING ▶️"

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Status: {status_text}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("Day 5 - Gesture Action", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
