import cv2
import mediapipe as mp
import pyautogui
import time
import math

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

paused = False
last_action = 0
cooldown = 1.2

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "NO HAND"

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark

            fingers = [
                finger_up(lm, 8, 6),   # index
                finger_up(lm, 12, 10), # middle
                finger_up(lm, 16, 14), # ring
                finger_up(lm, 20, 18)  # pinky
            ]

            thumb_up = lm[4].x > lm[3].x

            now = time.time()

            # STOP / RUN
            if fingers == [0,0,0,0]:
                paused = True
                gesture = "STOP ✊"

            elif fingers == [1,1,1,1]:
                paused = False
                gesture = "RUN ✋"

            # OPEN CHROME
            elif fingers == [1,1,0,0] and now-last_action > cooldown:
                pyautogui.press("win")
                time.sleep(0.4)
                pyautogui.write("chrome")
                pyautogui.press("enter")
                last_action = now
                gesture = "OPEN CHROME ✌️"

            # VOLUME UP
            elif thumb_up and fingers == [0,0,0,0] and now-last_action > cooldown:
                pyautogui.press("volumeup")
                last_action = now
                gesture = "VOLUME UP 👍"

            # VOLUME DOWN
            elif not thumb_up and fingers == [0,0,0,0] and now-last_action > cooldown:
                pyautogui.press("volumedown")
                last_action = now
                gesture = "VOLUME DOWN 👎"

            # CLOSE APP
            elif thumb_up and fingers == [0,0,0,1] and now-last_action > cooldown:
                pyautogui.hotkey("alt", "f4")
                last_action = now
                gesture = "CLOSE APP 🤙"

            # MOUSE CONTROL
            if not paused:
                x = int(lm[8].x * screen_w)
                y = int(lm[8].y * screen_h)
                pyautogui.moveTo(x, y)

                # CLICK (PINCH)
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                tx, ty = int(lm[4].x * w), int(lm[4].y * h)
                dist = math.hypot(ix-tx, iy-ty)

                if dist < 40 and now-last_action > 0.8:
                    pyautogui.click()
                    last_action = now

        cv2.putText(frame, gesture, (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("DAY 7 - FULL HAND CONTROL", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
