import cv2
import mediapipe as mp
import pyautogui
import time

# Safety settings
pyautogui.FAILSAFE = True   
pyautogui.PAUSE = 0.01

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

finger_tips = [8, 12, 16, 20]

paused = False
last_click_time = 0
click_cooldown = 0.8  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
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

            # ---------- Finger states (index to pinky) ----------
            fingers = []
            for tip in finger_tips:
                if lm[tip].y < lm[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # ---------- Gesture logic ----------
            if fingers == [0, 0, 0, 0]:
                gesture = "FIST ✊"
                paused = True
            elif fingers == [1, 1, 1, 1]:
                gesture = "OPEN PALM ✋"
                paused = False
            elif fingers == [1, 1, 0, 0]:
                gesture = "V SIGN ✌️"
            else:
                gesture = "CONTROL"

            # ---------- Cursor control ----------
            if not paused:
                # Index finger tip
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)

                # Map to screen
                screen_x = int(lm[8].x * screen_w)
                screen_y = int(lm[8].y * screen_h)

                pyautogui.moveTo(screen_x, screen_y)

                # ---------- Click (pinch) ----------
                thumb_x, thumb_y = int(lm[4].x * w), int(lm[4].y * h)
                pinch_dist = ((ix - thumb_x)**2 + (iy - thumb_y)**2) ** 0.5

                now = time.time()
                if pinch_dist < 40 and (now - last_click_time) > click_cooldown:
                    pyautogui.click()
                    last_click_time = now

        cv2.putText(
            frame,
            f"Status: {'PAUSED' if paused else 'RUNNING'}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if paused else (0, 255, 0),
            2
        )

        cv2.imshow("Day 6 - Hand Mouse Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
